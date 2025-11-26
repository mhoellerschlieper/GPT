/***********************************************************************
*  transformer_block_v2.rs  –  Erweiterter Decoder-Block (kompakt)
***********************************************************************/
// transformer_block_v2.rs
use std::any::Any;
use serde::{Serialize, Deserialize};
use bincode::{Encode, Decode};
use ndarray::Array2;
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::MultiHeadAttention;
use crate::feed_forward_geglu::FeedForwardGeGLU;
use crate::layer_time2vec::Time2Vec;
use crate::utils::dropout_inplace;
use crate::llm::Layer;

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct TransformerBlockV2 {
    pub norm1:       LayerNorm,
    pub attention:   MultiHeadAttention,
    pub norm2:       LayerNorm,
    pub feedforward: FeedForwardGeGLU,
    pub time_embed:  Time2Vec,
    pub f_dropout:   f32,
}


impl TransformerBlockV2 {
    pub fn new(i_embed: usize, i_hidden: usize, i_heads: usize, f_dropout: f32) -> Self {
        TransformerBlockV2 {
            norm1:       LayerNorm::new(i_embed),
            attention:   MultiHeadAttention::new(i_embed, i_heads, f_dropout),
            norm2:       LayerNorm::new(i_embed),
            feedforward: FeedForwardGeGLU::new(i_embed, i_hidden, f_dropout),
            time_embed:  Time2Vec::new(i_embed),
            f_dropout,
        }
    }

    fn forward_step(&mut self, m_x: &Array2<f32>, d_timestamp: f32, i_step: usize) -> Array2<f32> {
        // Zeit-Features + PreNorm
        let m_x_time = self.time_embed.add_time_axis(m_x, d_timestamp);
        let m_norm1  = self.norm1.normalize(&m_x_time);

        // Attention + Residual
        let m_attn = self.attention.forward(&m_norm1, i_step);
        let mut m_y = &m_attn + m_x;
        dropout_inplace(&mut m_y, self.f_dropout);

        // FeedForward + Residual
        let m_norm2 = self.norm2.normalize(&m_y);
        let m_ff = self.feedforward.forward(&m_norm2);
        let mut m_z = &m_ff + &m_y;
        dropout_inplace(&mut m_z, self.f_dropout);
        m_z
    }
}

impl Layer for TransformerBlockV2 {
    fn layer_type(&self) -> &str { "TransformerBlockV2" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    // Hinweis: LLM ruft forward ohne Zeit/Step auf.
    // Wir nutzen Defaults (0.0, 0) fuer einfache Inferenz/Training.
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_step(input, 0.0, 0)
    }

    // Einfache Rueckleitung (kein Update hier). Sicher und stabil.
    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        grads.clone()
    }

    fn parameters(&self) -> usize {
        // Schaetzung: Parameter zaehlen ist optional, hier 0
        0
    }
}
