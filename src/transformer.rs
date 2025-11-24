// =============================================
// transformers.rs
// =============================================
use ndarray::Array2;
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;


use crate::{
    feed_forward::FeedForward, layer_norm::LayerNorm, llm::Layer, self_attention::SelfAttention,
};
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct TransformerBlock {
    #[bincode(with_serde)] attention:     SelfAttention,
    #[bincode(with_serde)] feed_forward:  FeedForward,
    #[bincode(with_serde)] norm1:         LayerNorm,
    #[bincode(with_serde)] norm2:         LayerNorm,
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Standard Transformer architecture: attention + norm -> feedforward + norm
        let attention_out = self.attention.forward(input); // includes residual
        let norm1_out = self.norm1.normalize(&attention_out);

        let feed_forward_out = self.feed_forward.forward(&norm1_out); // includes residual

        self.norm2.normalize(&feed_forward_out)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backward through second LayerNorm
        let grad_norm2 = self.norm2.backward(grads, lr);

        // Backward through feed-forward (includes residual connection)
        let grad_ffn = self.feed_forward.backward(&grad_norm2, lr);

        // Backward through first LayerNorm
        let grad_norm1 = self.norm1.backward(&grad_ffn, lr);

        // Backward through attention (includes residual connection)

        self.attention.backward(&grad_norm1, lr)
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
