/***********************************************************************
*  feed_forward_geglu.rs   –  Gated Linear Unit-Variante (Fix)
***********************************************************************/
// feed_forward_geglu.rs
use serde::{Serialize, Deserialize};
use bincode::{Encode, Decode};
use ndarray::{Array2, Axis};
use crate::utils::{apply_weight_decay, dropout_inplace};

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct FeedForwardGeGLU {
    #[bincode(with_serde)]
    pub w_in:  Array2<f32>,   // embed x (2*hidden)
    #[bincode(with_serde)]
    pub w_out: Array2<f32>,   // hidden x embed
    pub f_dropout: f32,
    pub f_decay:   f32,
}

// restlicher Code unverändert


impl FeedForwardGeGLU {
    pub fn new(i_embed: usize, i_hidden: usize, f_dropout: f32) -> Self {
        assert!(i_embed > 0 && i_hidden > 0, "Dimensionen müssen > 0 sein");
        FeedForwardGeGLU {
            w_in:   Array2::<f32>::ones((i_embed, i_hidden * 2)) * 0.01,
            w_out:  Array2::<f32>::ones((i_hidden, i_embed))     * 0.01,
            f_dropout,
            f_decay: 1e-4,
        }
    }

    pub fn forward(&mut self, m_x: &Array2<f32>) -> Array2<f32> {
        let m_proj = m_x.dot(&self.w_in);

        let cols = m_proj.dim().1;
        assert!(cols % 2 == 0, "Proj-Spaltenzahl muss durch 2 teilbar sein");

        let (m_lin, m_gate) = m_proj.view().split_at(Axis(1), cols / 2);

        let m_gate_gelu = m_gate.mapv(gelu);
        let m_act = &m_lin * &m_gate_gelu;

        let mut m_out = m_act.dot(&self.w_out);

        dropout_inplace(&mut m_out, self.f_dropout);
        apply_weight_decay(&mut self.w_in,  self.f_decay);
        apply_weight_decay(&mut self.w_out, self.f_decay);

        m_out
    }
}

fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}
