/***********************************************************************
*  multi_head_attention.rs (Fix)
***********************************************************************/
// multi_head_attention.rs
use crate::layer_pos_encoding::apply_rope;
use crate::utils::{apply_weight_decay, dropout_inplace};
use bincode::{Decode, Encode};
use ndarray::{Array2, Axis, concatenate, s};
use serde::{Deserialize, Serialize};

use ndarray::parallel::prelude::*;
use rayon::prelude::*;

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
    pub w_qkv: Array2<f32>,
    #[bincode(with_serde)]
    pub w_o: Array2<f32>,

    // Runtime Cache: nicht serialisieren
    #[serde(skip)]
    #[bincode(with_serde)]
    v_cache_k: Option<Vec<Array2<f32>>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    v_cache_v: Option<Vec<Array2<f32>>>,

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
        let i_head_dim = i_embed / i_heads;

        MultiHeadAttention {
            i_embed,
            i_heads,
            i_head_dim,
            w_qkv: Array2::<f32>::from_elem((i_embed, i_embed * 3), 0.01),
            w_o: Array2::<f32>::from_elem((i_embed, i_embed), 0.01),
            v_cache_k: None,
            v_cache_v: None,
            f_dropout,
            f_decay: 1e-4,
        }
    }

    pub fn clear_cache(&mut self) {
        self.v_cache_k = Some(Vec::new());
        self.v_cache_v = Some(Vec::new());
    }

   
    pub fn forward(&mut self, m_x: &Array2<f32>, i_step: usize) -> Array2<f32> {
        // 1) Lineare Projektionen
        let m_qkv = m_x.dot(&self.w_qkv); // (seq, 3*embed)
        let (m_q_view, tmp) = m_qkv.view().split_at(Axis(1), self.i_embed);
        let (m_k_view, m_v_view) = tmp.split_at(Axis(1), self.i_embed);

        let mut m_q = m_q_view.to_owned();
        let mut m_k = m_k_view.to_owned();
        let m_v = m_v_view.to_owned();

        // 2) RoPE
        apply_rope(m_q.view_mut(), m_k.view_mut(), i_step, self.i_head_dim);

        // 3) Cache (optional)
        let v_k = self.v_cache_k.get_or_insert_with(Vec::new);
        let v_v = self.v_cache_v.get_or_insert_with(Vec::new);
        v_k.push(m_k.clone());
        v_v.push(m_v.clone());

        // 4) Heads trennen
        let q_heads = self.split_heads_vec(&m_q);
        let k_heads = self.split_heads_vec(&m_k);
        let v_heads = self.split_heads_vec(&m_v);

        // 5) Attention pro Head
        let scale = (self.i_head_dim as f32).sqrt();
        /*
        let mut ctx_per_head: Vec<Array2<f32>> = Vec::with_capacity(self.i_heads);
        for h in 0..self.i_heads {
            let mut scores = q_heads[h].dot(&k_heads[h].t()); // (seq, seq)
            scores.mapv_inplace(|v| v / scale);

            let probs = softmax_rows(&scores); // (seq, seq)
            let mut probs_dropped = probs.clone();
            dropout_inplace(&mut probs_dropped, self.f_dropout);

            let ctx = probs_dropped.dot(&v_heads[h]); // (seq, head_dim)
            ctx_per_head.push(ctx);
        }*/
        let ctx_per_head: Vec<Array2<f32>> = (0..self.i_heads)
            .into_par_iter()
            .map(|h| {
                let mut scores = q_heads[h].dot(&k_heads[h].t()); // (seq, seq)
                scores.mapv_inplace(|v| v / scale);
                let probs = softmax_rows_par(&scores); // parallel-softmax (unten)
                let mut probs_dropped = probs.clone();
                dropout_inplace(&mut probs_dropped, self.f_dropout); // thread-safe: nutzt thread_rng()
                probs_dropped.dot(&v_heads[h]) // (seq, head_dim)
            })
            .collect();

        // 6) Heads mergen: (seq, embed)
        let m_concat = self.merge_heads_vec(&ctx_per_head);

        // 7) Projektion + Decay
        let m_out = m_concat.dot(&self.w_o);
        apply_weight_decay(&mut self.w_qkv, self.f_decay);
        apply_weight_decay(&mut self.w_o, self.f_decay);

        m_out
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

fn softmax_rows(m: &Array2<f32>) -> Array2<f32> {
    let mut out = m.clone();
    for mut row in out.axis_iter_mut(Axis(0)) {
        let mut max_v = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    out
}
