/***********************************************************************
*  transformer_block_v2.rs  –  Alle gewünschten Erweiterungen
***********************************************************************/
use crate::{
    layer_norm::LayerNorm,
    multi_head_attention::MultiHeadAttention,
    feed_forward_geglu::FeedForwardGeGLU,
    layer_time2vec::Time2Vec,
    utils::dropout_inplace,
};
use ndarray::Array2;

pub struct TransformerBlockV2 {
    norm1:        LayerNorm,
    attention:    MultiHeadAttention,
    norm2:        LayerNorm,
    feedforward:  FeedForwardGeGLU,
    time_embed:   Time2Vec,
    f_dropout:    f32,
}

impl TransformerBlockV2 {
    pub fn new(i_embed: usize,
               i_hidden: usize,
               i_heads: usize,
               f_dropout: f32) -> Self
    {
        TransformerBlockV2 {
            norm1:       LayerNorm::new(i_embed),
            attention:   MultiHeadAttention::new(i_embed, i_heads, f_dropout),
            norm2:       LayerNorm::new(i_embed),
            feedforward: FeedForwardGeGLU::new(i_embed, i_hidden, f_dropout),
            time_embed:  Time2Vec::new(i_embed),
            f_dropout,
        }
    }

    /// `d_timestamp` = kontinuierliche Zeitvariable (z. B. Sekunden seit Epoch)
    /// `i_step`      = Positionsindex innerhalb der Sequenz
    pub fn forward(&mut self,
                   m_x: &Array2<f32>,   // Typ ergänzt
                   d_timestamp: f32,
                   i_step: usize) -> Array2<f32>  // Typ ergänzt
    {
        // ===== Pre-Norm + Zeitspur ======================================
        let m_x_time = self.time_embed.add_time_axis(m_x, d_timestamp);
        let m_norm1  = self.norm1.normalize(&m_x_time);

        // ===== Attention ===============================================
        let m_attn   = self.attention.forward(&m_norm1, i_step);
        let mut m_y  = &m_attn + m_x;                        // Residual
        dropout_inplace(&mut m_y, self.f_dropout);

        // ===== Feed-Forward ============================================
        let m_norm2 = self.norm2.normalize(&m_y);
        let m_ff    = self.feedforward.forward(&m_norm2);
        let mut m_z = &m_ff + &m_y;                          // Residual
        dropout_inplace(&mut m_z, self.f_dropout);

        m_z
    }
}
