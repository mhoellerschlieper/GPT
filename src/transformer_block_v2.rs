// src/transformer_block_v2.rs
// TransformerBlockV2 – korrekter Backward mit Residuals und Dropout-Masken
use crate::feed_forward_geglu::FeedForwardGeGLU;
use crate::layer_norm::LayerNorm;
use crate::layer_time2vec::Time2Vec;
use crate::llm::Layer;
use crate::multi_head_attention::MultiHeadAttention;
use bincode::{Decode, Encode};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::any::Any;
use rand::Rng;

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct TransformerBlockV2 {
    pub norm1: LayerNorm,
    pub attention: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub feedforward: FeedForwardGeGLU,
    pub time_embed: Time2Vec,
    pub f_dropout: f32,
    // Dropout-Masken (nur Runtime)
    #[serde(skip)]
    #[bincode(with_serde)]
    drop_mask_y: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    drop_mask_z: Option<Array2<f32>>,
}

impl TransformerBlockV2 {
    pub fn new(i_embed: usize, i_hidden: usize, i_heads: usize, f_dropout: f32) -> Self {
        TransformerBlockV2 {
            norm1: LayerNorm::new(i_embed),
            attention: MultiHeadAttention::new(i_embed, i_heads, f_dropout),
            norm2: LayerNorm::new(i_embed),
            feedforward: FeedForwardGeGLU::new(i_embed, i_hidden, f_dropout),
            time_embed: Time2Vec::new(i_embed),
            f_dropout,
            drop_mask_y: None,
            drop_mask_z: None,
        }
    }

    fn apply_dropout_local(&self, m: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
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

    fn forward_step(&mut self, m_x: &Array2<f32>, d_timestamp: f32, i_step: usize) -> Array2<f32> {
        // Zeit-Features + PreNorm
        let m_x_time = self.time_embed.add_time_axis(m_x, d_timestamp);
        let m_norm1 = self.norm1.normalize(&m_x_time);

        // Attention + Residual
        let m_attn = self.attention.forward(&m_norm1, i_step);
        let m_y = &m_attn + m_x; // residual 1
        let (mut m_y_drop, mask_y) = self.apply_dropout_local(&m_y);
        self.drop_mask_y = mask_y;

        // FeedForward + Residual
        let m_norm2 = self.norm2.normalize(&m_y_drop);
        let m_ff = self.feedforward.forward(&m_norm2);
        let m_z = &m_ff + &m_y_drop; // residual 2
        let (m_z_drop, mask_z) = self.apply_dropout_local(&m_z);
        self.drop_mask_z = mask_z;

        m_z_drop
    }
}

impl Layer for TransformerBlockV2 {
    fn layer_type(&self) -> &str {
        "TransformerBlockV2"
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.feedforward.parameter_count()
            + self.norm1.parameter_count()
            + self.norm2.parameter_count()
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_step(input, 0.0, 0)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 1) z_drop -> z
        let mut grad_z = grads.clone();
        if let Some(mask) = self.drop_mask_z.as_ref() {
            // rückwärts durch Dropout 2
            grad_z = &grad_z * mask;
        }

        // z = ff(norm2(y_drop)) + y_drop
        let grad_y_from_residual2 = grad_z.clone();  // Pfad der Residual-Verbindung
        let grad_ff_out = grad_z;                    // durch FFN-Zweig

        // durch FFN (liefert grad wrt norm2(y_drop))
        let grad_norm2_in = self.feedforward.backward(&grad_ff_out, lr);

        // Summe der beiden Wege vor Norm2
        let mut grad_y_drop = &grad_norm2_in + &grad_y_from_residual2;

        // durch Norm2
        grad_y_drop = self.norm2.backward(&grad_y_drop, lr);

        // 2) y_drop -> y
        if let Some(mask) = self.drop_mask_y.as_ref() {
            grad_y_drop = &grad_y_drop * mask; // rückwärts durch Dropout 1
        }

        // y = attn(norm1(x_time)) + x
        let grad_x_from_residual1 = grad_y_drop.clone();
        // durch Attention (liefert grad wrt norm1(x_time))
        let grad_attn_in = self.attention.backward(&grad_y_drop, lr);
        let mut grad_norm1_out = grad_attn_in + grad_x_from_residual1;

        // durch Norm1
        grad_norm1_out = self.norm1.backward(&grad_norm1_out, lr);

        // x_time = x + time2vec(t)  (time2vec ist statisch, nur Durchleitung)
        let grad_x = grad_norm1_out;
        grad_x
    }

    fn parameters(&self) -> usize {
        // historisch vorhanden; nicht genutzt
        0
    }
}
