// src/multi_head_attention.rs
// MultiHeadAttention – jetzt mit Adam, Backward und Dropout-Masken
use crate::adam::Adam;
use crate::layer_pos_encoding::{apply_rope, apply_rope_backward};
use crate::utils::apply_weight_decay;
use bincode::{Decode, Encode};
use ndarray::linalg::Dot;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis, Zip, concatenate, s};
use rand_distr::Normal;
use rand_distr::Distribution;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize}; // NEU: stellt .dot(...) sicher bereit

#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub enum AttentionError {
    EmptyCache,
}

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct MultiHeadAttention {
    pub i_embed: usize,
    pub i_heads: usize,
    pub i_head_dim: usize,
    #[bincode(with_serde)]
    pub w_qkv: Array2<f32>, // [embed, 3*embed]
    #[bincode(with_serde)]
    pub w_o: Array2<f32>, // [embed, embed]
    // Optimizer
    #[bincode(with_serde)]
    optimizer_qkv: Adam,
    #[bincode(with_serde)]
    optimizer_o: Adam,
    // Runtime Caches (nicht serialisieren)
    #[serde(skip)]
    #[bincode(with_serde)]
    v_cache_k: Option<Vec<Array2<f32>>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    v_cache_v: Option<Vec<Array2<f32>>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_dropout_masks: Option<Vec<Array2<f32>>>, // pro Head: [seq, seq]
    pub f_dropout: f32,
    pub f_decay: f32,
}

pub fn softmax_rows_par(m: &Array2<f32>) -> Array2<f32> {
    let mut out = m.clone();
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        });
    out
}

impl MultiHeadAttention {
    pub fn new(i_embed: usize, i_heads: usize, f_dropout: f32) -> Self {
        assert!(i_heads > 0, "i_heads muss > 0 sein");
        assert!(
            i_embed % i_heads == 0,
            "i_embed muss durch i_heads teilbar sein"
        );


        let mut rng = rand::thread_rng();
        let std_qkv = (2.0 / i_embed as f32).sqrt();
        let normal_qkv = Normal::new(0.0, std_qkv).unwrap();
        let std_o = (2.0 / i_embed as f32).sqrt();
        let normal_o = Normal::new(0.0, std_o).unwrap();


        let i_head_dim = i_embed / i_heads;
        MultiHeadAttention {
            i_embed,
            i_heads,
            i_head_dim,
            w_qkv: ndarray::Array2::from_shape_fn((i_embed, i_embed * 3), |_| normal_qkv.sample(&mut rng)),
            w_o:   ndarray::Array2::from_shape_fn((i_embed, i_embed),      |_| normal_o.sample(&mut rng)),
            optimizer_qkv: Adam::new((i_embed, i_embed * 3)),
            optimizer_o: Adam::new((i_embed, i_embed)),
            v_cache_k: None,
            v_cache_v: None,
            cached_input: None,
            cached_dropout_masks: None,
            f_dropout,
            f_decay: 1e-4,
        }
    }

    pub fn set_accumulate_steps(&mut self, steps: usize) {
        self.optimizer_qkv.set_accumulate_steps(steps);
        self.optimizer_o.set_accumulate_steps(steps);
    }

    pub fn parameter_count(&self) -> usize {
        self.w_qkv.len() + self.w_o.len()
    }

    pub fn clear_cache(&mut self) {
        self.v_cache_k = None;
        self.v_cache_v = None;
    }

    // eigenes Dropout mit Masken (inverted) für Attentions-Weights
    fn apply_dropout_probs(&self, m: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
        let p = self.f_dropout.clamp(0.0, 1.0);
        if p <= 0.0 {
            return (m.clone(), None);
        }
        let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
        let mut rng = rand::thread_rng();
        let mut out = m.clone();
        let mut mask = m.clone();
        for (o, mk) in out.iter_mut().zip(mask.iter_mut()) {
            let drop = rng.r#gen::<f32>() < p;
            if drop {
                *o = 0.0;
                *mk = 0.0;
            } else {
                *o *= scale;
                *mk = scale;
            }
        }
        (out, Some(mask))
    }

    pub fn forward(&mut self, m_x: &Array2<f32>, i_step: usize) -> Array2<f32> {
        // Cache-Handling: im Training (i_step==0) kein KV-Cache
        if i_step == 0 {
            self.v_cache_k = None;
            self.v_cache_v = None;
        }

        self.cached_input = Some(m_x.clone());

        // 1) QKV
        let m_qkv = m_x.dot(&self.w_qkv); // [seq, 3*embed]
        let (m_q_view, tmp) = m_qkv.view().split_at(Axis(1), self.i_embed);
        let (m_k_view, m_v_view) = tmp.split_at(Axis(1), self.i_embed);
        let mut m_q = m_q_view.to_owned();
        let mut m_k = m_k_view.to_owned();
        let m_v = m_v_view.to_owned();

        // 2) RoPE
        apply_rope(m_q.view_mut(), m_k.view_mut(), i_step, self.i_head_dim);

        // 3) optional KV-Cache für Inferenz
        if i_step > 0 {
            let v_k = self.v_cache_k.get_or_insert_with(Vec::new);
            let v_v = self.v_cache_v.get_or_insert_with(Vec::new);
            v_k.push(m_k.clone());
            v_v.push(m_v.clone());
        }

        // 4) Heads
        let q_heads = self.split_heads_vec(&m_q);
        let k_heads = self.split_heads_vec(&m_k);
        let v_heads = self.split_heads_vec(&m_v);

        // 5) Attention pro Head (mit Dropout-Masken speichern)
        let scale = (self.i_head_dim as f32).sqrt();

        let per_head: Vec<(Array2<f32>, Array2<f32>)> = (0..self.i_heads)
            .into_par_iter()
            .map(|h| {
                // scores: [seq, seq]
                let mut scores = q_heads[h].dot(&k_heads[h].t()); // [seq, seq]
                scores.mapv_inplace(|v| v / scale);
                // causal mask
                let seq = scores.nrows();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[(i, j)] = f32::NEG_INFINITY;
                    }
                }
                let probs = softmax_rows_par(&scores);

                // Dropout der Attention-Weights
                let (probs_dropped, mask_opt) = Self::dropout_probs_static(&probs, self.f_dropout);
                let mask_mat =
                    mask_opt.unwrap_or_else(|| ndarray::Array2::from_elem(probs.dim(), 1.0));

                // Kontext: [seq, head_dim]
                let ctx = probs_dropped.dot(&v_heads[h]);
                (ctx, mask_mat)
            })
            .collect();

        // ctx-Listen und Masken trennen
        let (ctx_per_head, masks): (Vec<Array2<f32>>, Vec<Array2<f32>>) =
            per_head.into_iter().unzip();
        self.cached_dropout_masks = Some(masks);

        // 6) Merge heads
        let m_concat = self.merge_heads_vec(&ctx_per_head); // [seq, embed]

        // 7) Projektion + Decay
        let m_out = m_concat.dot(&self.w_o);
        apply_weight_decay(&mut self.w_qkv, self.f_decay);
        apply_weight_decay(&mut self.w_o, self.f_decay);
        m_out
    }

    // kleiner Helper für rayon-Closure (kein &self)
    fn dropout_probs_static(m: &Array2<f32>, f_dropout: f32) -> (Array2<f32>, Option<Array2<f32>>) {
        let p = f_dropout.clamp(0.0, 1.0);
        if p <= 0.0 {
            return (m.clone(), None);
        }
        let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
        let mut rng = rand::thread_rng();
        let mut out = m.clone();
        let mut mask = m.clone();
        for (o, mk) in out.iter_mut().zip(mask.iter_mut()) {
            let drop = rng.r#gen::<f32>() < p;
            if drop {
                *o = 0.0;
                *mk = 0.0;
            } else {
                *o *= scale;
                *mk = scale;
            }
        }
        (out, Some(mask))
    }

    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Recompute Forward intermediates deterministisch
        let x = self.cached_input.as_ref().expect("forward zuerst"); // [seq, embed]

        // QKV und RoPE erneut
        let m_qkv = x.dot(&self.w_qkv);
        let (m_q_view, tmp) = m_qkv.view().split_at(Axis(1), self.i_embed);
        let (m_k_view, m_v_view) = tmp.split_at(Axis(1), self.i_embed);
        let mut q = m_q_view.to_owned();
        let mut k = m_k_view.to_owned();
        let v = m_v_view.to_owned();
        // Hinweis: Training nutzt i_step=0
        apply_rope(q.view_mut(), k.view_mut(), 0, self.i_head_dim);

        let q_heads = self.split_heads_vec(&q);
        let k_heads = self.split_heads_vec(&k);
        let v_heads = self.split_heads_vec(&v);

        // Vorwärts: ctx_per_head -> concat -> w_o
        // Backward: durch W_o
        let (seq, embed) = grads.dim();
        let heads = self.i_heads;
        let head_dim = self.i_head_dim;

        // m_concat aus Heads rekonstruieren
        let ctx_heads_forward: Vec<Array2<f32>> = (0..heads)
            .map(|h| {
                let scale = (head_dim as f32).sqrt();
                let mut scores = q_heads[h].dot(&k_heads[h].t());
                scores.mapv_inplace(|val| val / scale);
                // causal mask
                let seq = scores.nrows();
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[(i, j)] = f32::NEG_INFINITY;
                    }
                }
                let probs = softmax_rows_par(&scores);
                // Dropout-Maske aus Cache
                let mask = self
                    .cached_dropout_masks
                    .as_ref()
                    .and_then(|v| v.get(h))
                    .cloned()
                    .unwrap_or_else(|| Array2::<f32>::ones(probs.dim()));
                let probs_dropped = probs.clone() * &mask;
                probs_dropped.dot(&v_heads[h])
            })
            .collect();

        let m_concat = self.merge_heads_vec(&ctx_heads_forward); // [seq, embed]

        // Grad W_o und grad nach concat
        let grad_w_o = m_concat.t().dot(grads); // [embed, embed]
        let grad_concat = grads.dot(&self.w_o.t()); // [seq, embed]

        // split grad_concat zurück auf Heads
        let mut grad_ctx_heads: Vec<Array2<f32>> = Vec::with_capacity(heads);
        for h in 0..heads {
            let c0 = h * head_dim;
            let c1 = c0 + head_dim;
            grad_ctx_heads.push(grad_concat.slice(s![.., c0..c1]).to_owned());
        }

        // pro Head backward
        let mut grad_q = Array2::<f32>::zeros((seq, embed));
        let mut grad_k = Array2::<f32>::zeros((seq, embed));
        let mut grad_v = Array2::<f32>::zeros((seq, embed));

        for h in 0..heads {
            let qh = &q_heads[h]; // [seq, head_dim]
            let kh = &k_heads[h];
            let vh = &v_heads[h];

            // scores und probs erneut
            let scale = (head_dim as f32).sqrt();
            let mut scores = qh.dot(&kh.t()); // [seq, seq]
            scores.mapv_inplace(|val| val / scale);
            let probs = softmax_rows_par(&scores); // [seq, seq]

            // Dropout-Maske anwenden auf grad
            let mask = self
                .cached_dropout_masks
                .as_ref()
                .and_then(|v| v.get(h))
                .cloned()
                .unwrap_or_else(|| Array2::<f32>::ones(probs.dim()));

            // dL/dP_drop = dL/dCtx * V^T
            let grad_ctx = &grad_ctx_heads[h]; // [seq, head_dim]
            let grad_p_drop = grad_ctx.dot(&vh.t()); // [seq, seq]
            let grad_p = grad_p_drop * &mask; // [seq, seq]

            // softmax backward: zeilenweise
            let grad_scores = softmax_backward_rows(&probs, &grad_p); // [seq, seq]

            // dQ und dK
            let dq_h = grad_scores.dot(kh); // [seq, head_dim]
            let dk_h = grad_scores.t().dot(qh); // [seq, head_dim]
            let dv_h = probs.t().dot(&grad_ctx.view()); // [seq, head_dim]

            // in Sammelmatrizen einsetzen
            let mut gq_slice = grad_q.slice_mut(s![.., h * head_dim..(h + 1) * head_dim]);
            gq_slice.assign(&dq_h);
            let mut gk_slice = grad_k.slice_mut(s![.., h * head_dim..(h + 1) * head_dim]);
            gk_slice.assign(&dk_h);
            let mut gv_slice = grad_v.slice_mut(s![.., h * head_dim..(h + 1) * head_dim]);
            gv_slice.assign(&dv_h);
        }

        // RoPE Backward (inverse Rotation)
        apply_rope_backward(grad_q.view_mut(), grad_k.view_mut(), 0, head_dim);

        // Gradienten für W_qkv und X
        let wq = self.w_qkv.slice(s![.., 0..embed]).to_owned();
        let wk = self.w_qkv.slice(s![.., embed..2 * embed]).to_owned();
        let wv = self.w_qkv.slice(s![.., 2 * embed..3 * embed]).to_owned();

        let grad_w_q = x.t().dot(&grad_q);
        let grad_w_k = x.t().dot(&grad_k);
        let grad_w_v = x.t().dot(&grad_v);
        // zusammen
        let grad_w_qkv = concatenate(
            Axis(1),
            &[grad_w_q.view(), grad_w_k.view(), grad_w_v.view()],
        )
        .expect("concat grad_w_qkv fehlgeschlagen");

        let grad_x = grad_q.dot(&wq.t()) + grad_k.dot(&wk.t()) + grad_v.dot(&wv.t());

        // Optimizer mit Akkumulation
        self.optimizer_o.step(&mut self.w_o, &grad_w_o, lr);
        self.optimizer_qkv.step(&mut self.w_qkv, &grad_w_qkv, lr);

        grad_x
    }

    fn split_heads_vec(&self, m: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut out = Vec::with_capacity(self.i_heads);
        for h in 0..self.i_heads {
            let c0 = h * self.i_head_dim;
            let c1 = c0 + self.i_head_dim;
            out.push(m.slice(s![.., c0..c1]).to_owned());
        }
        out
    }

    fn merge_heads_vec(&self, parts: &[Array2<f32>]) -> Array2<f32> {
        let views: Vec<_> = parts.iter().map(|a| a.view()).collect();
        concatenate(Axis(1), &views).expect("concatenate heads failed")
    }
}

// Softmax-Backward zeilenweise
fn softmax_backward_rows(softmax_out: &Array2<f32>, grad_out: &Array2<f32>) -> Array2<f32> {
    let mut grad_in = Array2::<f32>::zeros(softmax_out.dim());
    for (((mut g_row), s_row), go_row) in grad_in
        .axis_iter_mut(Axis(0))
        .zip(softmax_out.axis_iter(Axis(0)))
        .zip(grad_out.axis_iter(Axis(0)))
    {
        let dot: f32 = s_row
            .iter()
            .zip(go_row.iter())
            .map(|(&y, &dy)| y * dy)
            .sum();
        for ((g, &y), &dy) in g_row.iter_mut().zip(s_row.iter()).zip(go_row.iter()) {
            *g = y * (dy - dot);
        }
    }
    grad_in
}
