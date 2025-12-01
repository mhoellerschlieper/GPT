// ============================================================================/
// layers.rs
// ============================================================================
// Autor:    Marcus Schlieper (ExpChat.ai)
// Kontakt:  mschlieper@ylook.de | Tel: 49 2338 8748862 | Mobil: 49 15115751864
// Firma:    ExpChat.ai – Der KI Chat Client fuer den Mittelstand
// Adresse:  Epscheider Str21, 58339 Breckerfeld
// Hinweis:  Alle Netzschichten (Layer) und das Layer-Trait.
//           Implementiert: Embeddings, LayerNorm, MultiHeadAttention,
//           FeedForwardGeGLU, OutputProjection, TransformerBlockV2.
//           Sichere Implementierung ohne unsafe.
// Historie:
//  - 2025-11-29: Datei komplett neu erstellt, Backprop-Korrekturen bei GeGLU
//                und Attention (Skalierung, Dropout, Causal Mask).
// ============================================================================

#![forbid(unsafe_code)]

use bincode::{Decode, Encode};
use ndarray::{Array2, Axis, Zip, concatenate, Slice};
use serde::{Deserialize, Serialize};
use std::any::Any;

use rand::Rng;

use crate::math::{
    Adam, apply_weight_decay,
    softmax_rows_par, softmax_backward_rows,
    apply_rope, apply_rope_backward,
};
use crate::utils::{Tokenizer, MAX_SEQ_LEN, EMBEDDING_DIM};

// ---------------------------------------------------------------------------
// Trait: Layer
// ---------------------------------------------------------------------------

pub trait Layer {
    fn layer_type(&self) -> &str;
    fn parameter_count(&self) -> usize;
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32>;
    
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    
    fn parameters(&self) -> usize { self.parameter_count() }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct Embeddings {
    #[bincode(with_serde)]
    pub token_embeddings: Array2<f32>,      // [vocab, embed]
    #[bincode(with_serde)]
    pub positional_embeddings: Array2<f32>, // [max_seq, embed]

    #[serde(skip)]
    #[bincode(with_serde)]
    pub cached_input: Option<Array2<f32>>,

    #[bincode(with_serde)]
    pub token_optimizer: Adam,
    #[bincode(with_serde)]
    pub positional_optimizer: Adam,
}

impl Embeddings {
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let i_vocab_size = tokenizer.vocab_size();
        Self {
            token_embeddings: Self::init_embeddings(i_vocab_size, EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((i_vocab_size, EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.token_optimizer.set_accumulate_steps(i_steps);
        self.positional_optimizer.set_accumulate_steps(i_steps);
    }

    fn init_embeddings(i_vocab_size: usize, i_embed: usize) -> Array2<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.02).expect("invalid normal");
        Array2::from_shape_fn((i_vocab_size, i_embed), |_| dist.sample(&mut rng))
    }

    fn init_positional_embeddings(i_max_seq: usize, i_embed: usize) -> Array2<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.02).expect("invalid normal");
        Array2::from_shape_fn((i_max_seq, i_embed), |_| dist.sample(&mut rng))
    }

    fn gather_tokens(m_embeddings: &Array2<f32>, v_token_ids: &[usize]) -> Array2<f32> {
        let i_embed = m_embeddings.ncols();
        let mut m_out = Array2::<f32>::zeros((v_token_ids.len(), i_embed));
        for (i_row, &i_tok) in v_token_ids.iter().enumerate() {
            assert!(i_tok < m_embeddings.nrows(), "token id out of bounds");
            m_out.row_mut(i_row).assign(&m_embeddings.row(i_tok));
        }
        m_out
    }

    fn slice_positions(m_pos: &Array2<f32>, i_seq: usize) -> Array2<f32> {
        assert!(
            i_seq <= m_pos.nrows(),
            "sequence length exceeds maximum: {} > {}",
            i_seq, m_pos.nrows()
        );
        m_pos.slice_axis(Axis(0), Slice::from(0..i_seq)).to_owned()
    }

    pub fn embed_tokens(&self, v_ids: &[usize]) -> Array2<f32> {
        let m_tok = Self::gather_tokens(&self.token_embeddings, v_ids);
        let m_pos = Self::slice_positions(&self.positional_embeddings, v_ids.len());
        m_tok + m_pos
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str { "Embeddings" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.token_embeddings.len() + self.positional_embeddings.len()
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input: [1, seq_len] mit Token-IDs als f32
        self.cached_input = Some(input.clone());
        let v_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&v_ids)
    }

    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let m_in = self.cached_input.as_ref().expect("forward first");
        let v_ids: Vec<usize> = m_in.iter().map(|&x| x as usize).collect();

        let mut m_grad_tok = Array2::<f32>::zeros(self.token_embeddings.dim());
        let mut m_grad_pos = Array2::<f32>::zeros(self.positional_embeddings.dim());

        for (i_row, &i_tok) in v_ids.iter().enumerate() {
            assert!(i_tok < self.token_embeddings.nrows(), "token id out of bounds");
            let grad_row = grads.row(i_row);
            {
                let mut row_tok = m_grad_tok.row_mut(i_tok);
                row_tok += &grad_row;
            }
            {
                let mut row_pos = m_grad_pos.row_mut(i_row);
                row_pos += &grad_row;
            }
        }

        self.token_optimizer.step(&mut self.token_embeddings, &m_grad_tok, d_lr);
        self.positional_optimizer.step(&mut self.positional_embeddings, &m_grad_pos, d_lr);

        // Gradient zum vorigen Layer: hier einfach durchreichen
        grads.to_owned()
    }

    fn parameters(&self) -> usize {
        self.token_embeddings.len() + self.positional_embeddings.len()
    }
}

// ---------------------------------------------------------------------------
// LayerNorm (Pre-Norm)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct LayerNorm {
    d_epsilon: f32,
    #[bincode(with_serde)]
    m_gamma: Array2<f32>, // [1, embed]
    #[bincode(with_serde)]
    m_beta: Array2<f32>,  // [1, embed]

    #[serde(skip)]
    #[bincode(with_serde)]
    m_cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    m_cached_mean: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    m_cached_denom: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    m_cached_x_hat: Option<Array2<f32>>,

    #[bincode(with_serde)]
    opt_gamma: Adam,
    #[bincode(with_serde)]
    opt_beta: Adam,
}

impl LayerNorm {
    pub fn new(i_embed: usize) -> Self {
        Self {
            d_epsilon: 1e-5,
            m_gamma: Array2::ones((1, i_embed)),
            m_beta: Array2::zeros((1, i_embed)),
            m_cached_input: None,
            m_cached_mean: None,
            m_cached_denom: None,
            m_cached_x_hat: None,
            opt_gamma: Adam::new((1, i_embed)),
            opt_beta: Adam::new((1, i_embed)),
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.opt_gamma.set_accumulate_steps(i_steps);
        self.opt_beta.set_accumulate_steps(i_steps);
    }

    pub fn parameter_count(&self) -> usize {
        self.m_gamma.len() + self.m_beta.len()
    }

    fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let m_mean = input.mean_axis(Axis(1)).expect("mean failed").insert_axis(Axis(1));
        let m_var = input.var_axis(Axis(1), 0.0).insert_axis(Axis(1));
        let m_denom = (m_var.mapv(|v| v + self.d_epsilon)).mapv(|v| v.sqrt());
        let m_x_hat = (input - &m_mean) / &m_denom;
        let m_out = &self.m_gamma * &m_x_hat + &self.m_beta;

        self.m_cached_input = Some(input.clone());
        self.m_cached_mean = Some(m_mean);
        self.m_cached_denom = Some(m_denom);
        self.m_cached_x_hat = Some(m_x_hat);

        m_out
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str { "LayerNorm" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.m_gamma.len() + self.m_beta.len()
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let m_x = self.m_cached_input.as_ref().expect("forward first");
        let m_denom = self.m_cached_denom.as_ref().expect("denom missing");
        let m_x_hat = self.m_cached_x_hat.as_ref().expect("x_hat missing");

        let m_grad_gamma = (m_x_hat * grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let m_grad_beta = grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let m_dy_gamma = grads * &self.m_gamma;

        let m_mean_dy = m_dy_gamma
            .mean_axis(Axis(1))
            .expect("mean failed")
            .insert_axis(Axis(1));
        let m_mean_dy_xhat = (m_dy_gamma.clone() * m_x_hat)
            .mean_axis(Axis(1))
            .expect("mean failed")
            .insert_axis(Axis(1));

        let m_dx = (m_dy_gamma - &m_mean_dy - m_x_hat * &m_mean_dy_xhat) / m_denom;

        self.opt_gamma.step(&mut self.m_gamma, &m_grad_gamma, d_lr);
        self.opt_beta.step(&mut self.m_beta, &m_grad_beta, d_lr);

        m_dx
    }

    fn parameters(&self) -> usize {
        self.m_gamma.len() + self.m_beta.len()
    }
}

// ---------------------------------------------------------------------------
// MultiHeadAttention (mit RoPE, Causal Mask, Dropout)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct MultiHeadAttention {
    pub i_embed: usize,
    pub i_heads: usize,
    pub i_head_dim: usize,

    #[bincode(with_serde)]
    pub w_qkv: Array2<f32>, // [embed, 3*embed]
    #[bincode(with_serde)]
    pub w_o: Array2<f32>,   // [embed, embed]

    #[bincode(with_serde)]
    opt_qkv: Adam,
    #[bincode(with_serde)]
    opt_o: Adam,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_dropout_masks: Option<Vec<Array2<f32>>>,

    pub f_dropout: f32,
    pub f_decay: f32,
}

impl MultiHeadAttention {
    pub fn new(i_embed: usize, i_heads: usize, f_dropout: f32) -> Self {
        assert!(i_heads > 0, "heads must be > 0");
        assert!(i_embed % i_heads == 0, "embed must be divisible by heads");

        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();

        let d_std_qkv = (2.0 / i_embed as f32).sqrt();
        let dist_qkv = Normal::new(0.0, d_std_qkv).unwrap();

        let d_std_o = (2.0 / i_embed as f32).sqrt();
        let dist_o = Normal::new(0.0, d_std_o).unwrap();

        let i_head_dim = i_embed / i_heads;

        Self {
            i_embed,
            i_heads,
            i_head_dim,
            w_qkv: Array2::from_shape_fn((i_embed, i_embed * 3), |_| dist_qkv.sample(&mut rng)),
            w_o: Array2::from_shape_fn((i_embed, i_embed), |_| dist_o.sample(&mut rng)),
            opt_qkv: Adam::new((i_embed, i_embed * 3)),
            opt_o: Adam::new((i_embed, i_embed)),
            cached_input: None,
            cached_dropout_masks: None,
            f_dropout,
            f_decay: 0.0001,
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.opt_qkv.set_accumulate_steps(i_steps);
        self.opt_o.set_accumulate_steps(i_steps);
    }

    pub fn parameter_count(&self) -> usize {
        self.w_qkv.len() + self.w_o.len()
    }

    fn dropout_probs(m_in: &Array2<f32>, f_dropout: f32) -> (Array2<f32>, Option<Array2<f32>>) {
        let d_p = f_dropout.clamp(0.0, 1.0);
        if d_p <= 0.0 {
            return (m_in.clone(), None);
        }
        let d_scale = if d_p < 1.0 { 1.0 / (1.0 - d_p) } else { 0.0 };
        let mut rng = rand::thread_rng();
        let mut m_out = m_in.clone();
        let mut m_mask = m_in.clone();
        for (o, mk) in m_out.iter_mut().zip(m_mask.iter_mut()) {
            let b_drop = rng.random::<f32>() < d_p;
            if b_drop {
                *o = 0.0;
                *mk = 0.0;
            } else {
                *o *= d_scale;
                *mk = d_scale;
            }
        }
        (m_out, Some(m_mask))
    }

    fn split_heads_vec(&self, m: &Array2<f32>) -> Vec<Array2<f32>> {
        assert_eq!(m.ncols(), self.i_embed, "split_heads: cols != embed");
        let mut v_out = Vec::with_capacity(self.i_heads);
        for i_h in 0..self.i_heads {
            let i_c0 = i_h * self.i_head_dim;
            let i_c1 = i_c0 + self.i_head_dim;
            assert!(i_c1 <= m.ncols(), "split_heads: slice end OOB");
            v_out.push(m.slice_axis(Axis(1), Slice::from(i_c0..i_c1)).to_owned());
        }
        v_out
    }

    fn merge_heads_vec(&self, v_parts: &[Array2<f32>]) -> Array2<f32> {
        let views: Vec<_> = v_parts.iter().map(|a| a.view()).collect();
        concatenate(Axis(1), &views).expect("concatenate heads failed")
    }
}

impl Layer for MultiHeadAttention {
    fn layer_type(&self) -> &str { "MultiHeadAttention" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.w_qkv.len() + self.w_o.len()
    }

    fn forward(&mut self, m_x: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(m_x.clone());

        // QKV
        let m_qkv = m_x.dot(&self.w_qkv);
        let (v0, v1) = m_qkv.view().split_at(Axis(1), self.i_embed);
        let (v2, v3) = v1.split_at(Axis(1), self.i_embed);
        let mut m_q = v0.to_owned();
        let mut m_k = v2.to_owned();
        let m_v = v3.to_owned();

        // RoPE
        apply_rope(m_q.view_mut(), m_k.view_mut(), 0, self.i_head_dim);

        let v_q = self.split_heads_vec(&m_q);
        let v_k = self.split_heads_vec(&m_k);
        let v_v = self.split_heads_vec(&m_v);

        let d_scale = (self.i_head_dim as f32).sqrt();
        let mut v_ctx: Vec<Array2<f32>> = Vec::with_capacity(self.i_heads);
        let mut v_masks: Vec<Array2<f32>> = Vec::with_capacity(self.i_heads);

        for i_h in 0..self.i_heads {
            // Scores
            let mut m_scores = v_q[i_h].dot(&v_k[i_h].t());
            m_scores.mapv_inplace(|val| val / d_scale);

            // Causal mask
            let i_seq = m_scores.nrows();
            for i in 0..i_seq {
                for j in (i + 1)..i_seq {
                    m_scores[(i, j)] = f32::NEG_INFINITY;
                }
            }

            // Softmax + Dropout
            let m_probs = softmax_rows_par(&m_scores);
            let (m_probs_drop, opt_mask) = Self::dropout_probs(&m_probs, self.f_dropout);
            let m_mask = opt_mask.unwrap_or_else(|| Array2::from_elem(m_probs.dim(), 1.0f32));

            // Kontext
            let m_ctx = m_probs_drop.dot(&v_v[i_h]);
            v_ctx.push(m_ctx);
            v_masks.push(m_mask);
        }

        self.cached_dropout_masks = Some(v_masks);

        let m_concat = self.merge_heads_vec(&v_ctx);
        let m_out = m_concat.dot(&self.w_o);

        // Optional: Weight Decay (deaktiviert standardmaessig)
        let mut wqkv = self.w_qkv.clone();
        let mut wo = self.w_o.clone();
        let _ = (&mut wqkv, &mut wo); // placeholder um ungenutzte Warnungen zu vermeiden
        // apply_weight_decay(&mut wqkv, self.f_decay);
        // apply_weight_decay(&mut wo, self.f_decay);
        // self.w_qkv = wqkv;
        // self.w_o = wo;

        m_out
    }

    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let m_x = self.cached_input.as_ref().expect("forward first");

        // Recompute Q, K, V
        let m_qkv = m_x.dot(&self.w_qkv);
        let (v0, v1) = m_qkv.view().split_at(Axis(1), self.i_embed);
        let (v2, v3) = v1.split_at(Axis(1), self.i_embed);
        let mut m_q = v0.to_owned();
        let mut m_k = v2.to_owned();
        let m_v = v3.to_owned();

        // RoPE
        apply_rope(m_q.view_mut(), m_k.view_mut(), 0, self.i_head_dim);

        let v_q = self.split_heads_vec(&m_q);
        let v_k = self.split_heads_vec(&m_k);
        let v_v = self.split_heads_vec(&m_v);

        let (i_seq, i_embed) = grads.dim();
        let i_heads = self.i_heads;
        let i_hdim = self.i_head_dim;

        // Vorwaerts-Kontext fuer dW_o
        let mut v_ctx_fwd: Vec<Array2<f32>> = Vec::with_capacity(i_heads);
        let mut v_probs_cache: Vec<Array2<f32>> = Vec::with_capacity(i_heads);
        let mut v_mask_cache: Vec<Array2<f32>> = Vec::with_capacity(i_heads);

        for i_h in 0..i_heads {
            let d_scale = (i_hdim as f32).sqrt();
            let mut m_scores = v_q[i_h].dot(&v_k[i_h].t());
            m_scores.mapv_inplace(|val| val / d_scale);

            // Causal mask
            let i_len = m_scores.nrows();
            for i in 0..i_len {
                for j in (i + 1)..i_len {
                    m_scores[(i, j)] = f32::NEG_INFINITY;
                }
            }

            let m_probs = softmax_rows_par(&m_scores);
            let m_mask = self
                .cached_dropout_masks
                .as_ref()
                .and_then(|vm| vm.get(i_h))
                .cloned()
                .unwrap_or_else(|| Array2::<f32>::ones(m_probs.dim()));

            let m_probs_drop = &m_probs * &m_mask;
            let m_ctx = m_probs_drop.dot(&v_v[i_h]);

            v_ctx_fwd.push(m_ctx);
            v_probs_cache.push(m_probs);
            v_mask_cache.push(m_mask);
        }

        // dW_o und dConcat
        let m_concat = self.merge_heads_vec(&v_ctx_fwd);
        let m_grad_w_o = m_concat.t().dot(grads);
        let m_grad_concat = grads.dot(&self.w_o.t());

        // Split grad fuer je Head
        let mut v_grad_ctx: Vec<Array2<f32>> = Vec::with_capacity(i_heads);
        for i_h in 0..i_heads {
            let i_c0 = i_h * i_hdim;
            let i_c1 = i_c0 + i_hdim;
            assert!(i_c1 <= m_grad_concat.ncols(), "grad concat slice OOB");
            v_grad_ctx.push(m_grad_concat.slice_axis(Axis(1), Slice::from(i_c0..i_c1)).to_owned());
        }

        // Gradienten fuer Q, K, V
        let mut m_grad_q = Array2::<f32>::zeros((i_seq, i_embed));
        let mut m_grad_k = Array2::<f32>::zeros((i_seq, i_embed));
        let mut m_grad_v = Array2::<f32>::zeros((i_seq, i_embed));

        for i_h in 0..i_heads {
            let qh = &v_q[i_h];
            let kh = &v_k[i_h];
            let vh = &v_v[i_h];

            let d_scale = (i_hdim as f32).sqrt();

            // Probs und Maske aus Cache
            let m_probs = &v_probs_cache[i_h];
            let m_mask = &v_mask_cache[i_h];
            let m_probs_drop = m_probs * m_mask;

            // Gradfluss
            let m_grad_ctx_h = &v_grad_ctx[i_h];

            // dL/dP_drop
            let m_grad_p_drop = m_grad_ctx_h.dot(&vh.t());
            // dL/dP (Maske zuruecknehmen)
            let m_grad_p = &m_grad_p_drop * m_mask;

            // dL/dScores (scaled)
            let m_grad_scores_scaled = softmax_backward_rows(m_probs, &m_grad_p);
            // Korrektur fuer Skalierung (scores = unscaled / scale)
            let m_grad_scores = m_grad_scores_scaled.mapv(|v| v / d_scale);

            // dQ, dK, dV
            let m_dq = m_grad_scores.dot(kh);
            let m_dk = m_grad_scores.t().dot(qh);
            let m_dv = m_probs_drop.t().dot(&m_grad_ctx_h.view());

            // In die Gesamtmatrizen zurueckschreiben
            {
                let mut sl_q = m_grad_q.slice_axis_mut(Axis(1), Slice::from(i_h * i_hdim..(i_h + 1) * i_hdim));
                sl_q.assign(&m_dq);
            }
            {
                let mut sl_k = m_grad_k.slice_axis_mut(Axis(1), Slice::from(i_h * i_hdim..(i_h + 1) * i_hdim));
                sl_k.assign(&m_dk);
            }
            {
                let mut sl_v = m_grad_v.slice_axis_mut(Axis(1), Slice::from(i_h * i_hdim..(i_h + 1) * i_hdim));
                sl_v.assign(&m_dv);
            }
        }

        // RoPE Rueckwaerts
        apply_rope_backward(m_grad_q.view_mut(), m_grad_k.view_mut(), 0, i_hdim);

        // W_qkv aufsplitten
        assert_eq!(self.w_qkv.ncols(), 3 * i_embed, "w_qkv cols mismatch");
        let wq = self.w_qkv.slice_axis(Axis(1), Slice::from(0..i_embed)).to_owned();
        let wk = self.w_qkv.slice_axis(Axis(1), Slice::from(i_embed..2 * i_embed)).to_owned();
        let wv = self.w_qkv.slice_axis(Axis(1), Slice::from(2 * i_embed..3 * i_embed)).to_owned();

        // Gradienten fuer W_qkv
        let m_grad_w_q = m_x.t().dot(&m_grad_q);
        let m_grad_w_k = m_x.t().dot(&m_grad_k);
        let m_grad_w_v = m_x.t().dot(&m_grad_v);
        let m_grad_w_qkv = concatenate(
            Axis(1),
            &[m_grad_w_q.view(), m_grad_w_k.view(), m_grad_w_v.view()],
        ).expect("concat grad w_qkv failed");

        // Grad fuer Input
        let m_grad_x = m_grad_q.dot(&wq.t()) + m_grad_k.dot(&wk.t()) + m_grad_v.dot(&wv.t());

        // Optimizer Steps
        self.opt_o.step(&mut self.w_o, &m_grad_w_o, d_lr);
        self.opt_qkv.step(&mut self.w_qkv, &m_grad_w_qkv, d_lr);

        m_grad_x
    }

    fn parameters(&self) -> usize {
        self.w_qkv.len() + self.w_o.len()
    }
}

// ---------------------------------------------------------------------------
// FeedForward GeGLU (Gated GELU)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct FeedForwardGeGLU {
    #[bincode(with_serde)]
    pub w_in: Array2<f32>,   // [embed, 2*hidden] -> split in lin|gate
    #[bincode(with_serde)]
    pub w_out: Array2<f32>,  // [hidden, embed]
    pub f_dropout: f32,
    pub f_decay: f32,

    #[bincode(with_serde)]
    opt_w_in: Adam,
    #[bincode(with_serde)]
    opt_w_out: Adam,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_lin: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_gate: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_gate_gelu: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_act: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_dropout_mask: Option<Array2<f32>>,
}

impl FeedForwardGeGLU {
    pub fn new(i_embed: usize, i_hidden: usize, f_dropout: f32) -> Self {
        assert!(i_embed > 0 && i_hidden > 0, "invalid dims");
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();

        let d_std_in = (2.0 / i_embed as f32).sqrt();
        let d_std_out = (2.0 / i_hidden as f32).sqrt();
        let dist_in = Normal::new(0.0, d_std_in).unwrap();
        let dist_out = Normal::new(0.0, d_std_out).unwrap();

        Self {
            w_in: Array2::from_shape_fn((i_embed, i_hidden * 2), |_| dist_in.sample(&mut rng)),
            w_out: Array2::from_shape_fn((i_hidden, i_embed), |_| dist_out.sample(&mut rng)),
            f_dropout,
            f_decay: 0.0001,
            opt_w_in: Adam::new((i_embed, i_hidden * 2)),
            opt_w_out: Adam::new((i_hidden, i_embed)),
            cached_input: None,
            cached_lin: None,
            cached_gate: None,
            cached_gate_gelu: None,
            cached_act: None,
            cached_dropout_mask: None,
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.opt_w_in.set_accumulate_steps(i_steps);
        self.opt_w_out.set_accumulate_steps(i_steps);
    }

    pub fn parameter_count(&self) -> usize {
        self.w_in.len() + self.w_out.len()
    }

    fn apply_dropout(&self, m_in: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
        let d_p = self.f_dropout.clamp(0.0, 1.0);
        if d_p <= 0.0 {
            return (m_in.clone(), None);
        }
        let d_scale = if d_p < 1.0 { 1.0 / (1.0 - d_p) } else { 0.0 };
        let mut rng = rand::thread_rng();
        let mut m_out = m_in.clone();
        let mut m_mask = m_in.clone();
        for (o, mk) in m_out.iter_mut().zip(m_mask.iter_mut()) {
            let b_drop = rng.random::<f32>() < d_p;
            if b_drop {
                *o = 0.0;
                *mk = 0.0;
            } else {
                *o *= d_scale;
                *mk = d_scale;
            }
        }
        (m_out, Some(m_mask))
    }

    pub fn forward(&mut self, m_x: &Array2<f32>) -> Array2<f32> {
        // Projektieren
        let m_proj = m_x.dot(&self.w_in);
        let i_cols = m_proj.ncols();
        assert!(i_cols % 2 == 0, "projection must split evenly");
        let (v_lin, v_gate) = m_proj.view().split_at(Axis(1), i_cols / 2);
        let m_lin = v_lin.to_owned();
        let m_gate = v_gate.to_owned();

        // GeGLU: lin * gelu(gate)
        let m_gate_gelu = m_gate.mapv(gelu);
        let m_act = &m_lin * &m_gate_gelu;

        let m_out_raw = m_act.dot(&self.w_out);
        let (m_out, opt_mask) = self.apply_dropout(&m_out_raw);

        // Optional: Weight Decay (standard deaktiviert)
        let mut w_in = self.w_in.clone();
        let mut w_out = self.w_out.clone();
        let _ = (&mut w_in, &mut w_out);
        // apply_weight_decay(&mut w_in, self.f_decay);
        // apply_weight_decay(&mut w_out, self.f_decay);
        // self.w_in = w_in;
        // self.w_out = w_out;

        // Cache
        self.cached_input = Some(m_x.clone());
        self.cached_lin = Some(m_lin);
        self.cached_gate = Some(m_gate.clone());
        self.cached_gate_gelu = Some(m_gate_gelu);
        self.cached_act = Some(m_act);
        self.cached_dropout_mask = opt_mask;

        m_out
    }

    pub fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let m_x = self.cached_input.as_ref().expect("forward first");
        let m_lin = self.cached_lin.as_ref().expect("lin missing");
        let m_gate = self.cached_gate.as_ref().expect("gate missing");
        let m_gate_gelu = self.cached_gate_gelu.as_ref().expect("gate gelu missing");
        let m_act = self.cached_act.as_ref().expect("act missing");

        // Ruecknahme Dropout
        let mut m_grad_after_drop = grads.clone();
        if let Some(m_mask) = self.cached_dropout_mask.as_ref() {
            Zip::from(&mut m_grad_after_drop).and(m_mask).for_each(|g, &m| *g = *g * m);
        }

        // dW_out, dAct
        let m_grad_w_out = m_act.t().dot(&m_grad_after_drop);
        let m_grad_act = m_grad_after_drop.dot(&self.w_out.t());

        // Kettenregel GeGLU: act = lin * gelu(gate)
        let m_grad_lin = &m_grad_act * m_gate_gelu;
        let m_grad_gate_gelu = &m_grad_act * m_lin;
        let m_dgelu_gate = m_gate.mapv(dgelu); // Ableitung an Voraktivierung
        let m_grad_gate = &m_grad_gate_gelu * &m_dgelu_gate;

        // Zusammenbauen fuer w_in
        let (i_seq, i_h) = (m_grad_lin.nrows(), m_grad_lin.ncols());
        let mut m_grad_proj = Array2::<f32>::zeros((i_seq, 2 * i_h));
        {
            let mut left = m_grad_proj.slice_axis_mut(Axis(1), Slice::from(0..i_h));
            left.assign(&m_grad_lin);
        }
        {
            let mut right = m_grad_proj.slice_axis_mut(Axis(1), Slice::from(i_h..2 * i_h));
            right.assign(&m_grad_gate);
        }

        let m_grad_w_in = m_x.t().dot(&m_grad_proj);
        let m_grad_x = m_grad_proj.dot(&self.w_in.t());

        // Optimizer Steps
        self.opt_w_out.step(&mut self.w_out, &m_grad_w_out, d_lr);
        self.opt_w_in.step(&mut self.w_in, &m_grad_w_in, d_lr);

        m_grad_x
    }
}

fn gelu(x: f32) -> f32 {
    const D_SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (D_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

fn dgelu(x: f32) -> f32 {
    const D_SQRT_2_OVER_PI: f32 = 0.7978845608;
    let x3 = x * x * x;
    let t = D_SQRT_2_OVER_PI * (x + 0.044715 * x3);
    let th = t.tanh();
    let sech2 = 1.0 - th * th;
    0.5 * (1.0 + th) + 0.5 * x * sech2 * D_SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
}

// ---------------------------------------------------------------------------
// OutputProjection (letzte Schicht)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct OutputProjection {
    #[bincode(with_serde)]
    pub w_out: Array2<f32>, // [embed, vocab]
    #[bincode(with_serde)]
    pub b_out: Array2<f32>, // [1, vocab]
    #[bincode(with_serde)]
    pub opt_w: Adam,
    #[bincode(with_serde)]
    pub opt_b: Adam,

    #[serde(skip)]
    #[bincode(with_serde)]
    pub cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    pub fn new(i_embed: usize, i_vocab: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let d_std = (2.0 / i_embed as f32).sqrt();
        let dist = Normal::new(0.0, d_std).expect("invalid normal");
        Self {
            w_out: Array2::from_shape_fn((i_embed, i_vocab), |_| dist.sample(&mut rng)),
            b_out: Array2::zeros((1, i_vocab)),
            opt_w: Adam::new((i_embed, i_vocab)),
            opt_b: Adam::new((1, i_vocab)),
            cached_input: None,
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.opt_w.set_accumulate_steps(i_steps);
        self.opt_b.set_accumulate_steps(i_steps);
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str { "OutputProjection" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.w_out.nrows(), "embed mismatch");
        assert_eq!(self.b_out.ncols(), self.w_out.ncols(), "bias size mismatch");
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        let m_in = self.cached_input.as_ref().expect("forward first");
        let m_grad_w = m_in.t().dot(grads);
        let m_grad_b = grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        let m_grad_in = grads.dot(&self.w_out.t());

        self.opt_w.step(&mut self.w_out, &m_grad_w, d_lr);
        self.opt_b.step(&mut self.b_out, &m_grad_b, d_lr);

        m_grad_in
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}

// ---------------------------------------------------------------------------
// TransformerBlockV2 (Pre-Norm, Residuals)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct TransformerBlockV2 {
    #[bincode(with_serde)]
    pub norm1: LayerNorm,
    #[bincode(with_serde)]
    pub attention: MultiHeadAttention,
    #[bincode(with_serde)]
    pub norm2: LayerNorm,
    #[bincode(with_serde)]
    pub feedforward: FeedForwardGeGLU,
}

impl TransformerBlockV2 {
    pub fn new(i_embed: usize, i_hidden: usize, i_heads: usize, f_dropout: f32) -> Self {
        Self {
            norm1: LayerNorm::new(i_embed),
            attention: MultiHeadAttention::new(i_embed, i_heads, f_dropout),
            norm2: LayerNorm::new(i_embed),
            feedforward: FeedForwardGeGLU::new(i_embed, i_hidden, f_dropout),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.norm1.parameter_count()
            + self.attention.parameter_count()
            + self.norm2.parameter_count()
            + self.feedforward.parameter_count()
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.norm1.set_accumulate_steps(i_steps);
        self.attention.set_accumulate_steps(i_steps);
        self.norm2.set_accumulate_steps(i_steps);
        self.feedforward.set_accumulate_steps(i_steps);
    }
}

impl Layer for TransformerBlockV2 {
    fn layer_type(&self) -> &str { "TransformerBlockV2" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.norm1.parameter_count()
            + self.attention.parameter_count()
            + self.norm2.parameter_count()
            + self.feedforward.parameter_count()
    }


    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Pre-Norm Attention
        let m_x1 = self.norm1.forward(input);
        let m_attn = self.attention.forward(&m_x1);
        let m_res1 = input + &m_attn;

        // Pre-Norm FFN
        let m_x2 = self.norm2.forward(&m_res1);
        let m_ff = self.feedforward.forward(&m_x2);
        let m_res2 = m_res1 + &m_ff;
        m_res2
    }

    fn backward(&mut self, grads: &Array2<f32>, d_lr: f32) -> Array2<f32> {
        // res2 = res1 + ff
        let m_grad_res1_skip = grads.clone();
        let m_grad_ff = grads.clone();

        // durch FF und Norm2
        let m_grad_x2 = self.feedforward.backward(&m_grad_ff, d_lr);
        let m_grad_res1_from_norm2 = self.norm2.backward(&m_grad_x2, d_lr);

        // summe an res1
        let m_grad_res1_total = m_grad_res1_skip + &m_grad_res1_from_norm2;

        // res1 = input + attn(x1)
        let m_grad_input_from_skip = m_grad_res1_total.clone();
        let m_grad_attn_input = self.attention.backward(&m_grad_res1_total, d_lr);

        // x1 = norm1(input)
        let m_grad_input_from_norm1 = self.norm1.backward(&m_grad_attn_input, d_lr);

        // gesamt
        m_grad_input_from_skip + &m_grad_input_from_norm1
    }

    fn parameters(&self) -> usize {
        self.parameter_count()
    }
}
