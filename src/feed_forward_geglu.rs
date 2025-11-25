/***********************************************************************
*  feed_forward_geglu.rs   –  Gated Linear Unit-Variante (Fix)
***********************************************************************/
use ndarray::{Array2, Axis};
use crate::utils::{apply_weight_decay, dropout_inplace};

pub struct FeedForwardGeGLU {
    // Gewichte
    w_in:  Array2<f32>,   // embed × (2*hidden): [lin | gate]
    w_out: Array2<f32>,   // hidden × embed
    // Regularisierung
    f_dropout: f32,
    f_decay:   f32,
}

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
        // Projektion: seq × (2*hidden)
        let m_proj = m_x.dot(&self.w_in);

        // Sicherheitscheck: gerade Spaltenzahl
        let cols = m_proj.dim().1;
        assert!(cols % 2 == 0, "Proj-Spaltenzahl muss durch 2 teilbar sein");

        // Aufteilen in [lin | gate]
        let (m_lin, m_gate) = m_proj.view().split_at(Axis(1), cols / 2);

        // GeGLU: lin * gelu(gate)
        let m_gate_gelu = m_gate.mapv(gelu);
        let m_act = &m_lin * &m_gate_gelu; // ergibt (seq × hidden)

        // Projektionsausgabe
        let mut m_out = m_act.dot(&self.w_out); // (seq × embed)

        // Dropout + Weight Decay
        dropout_inplace(&mut m_out, self.f_dropout);
        apply_weight_decay(&mut self.w_in,  self.f_decay);
        apply_weight_decay(&mut self.w_out, self.f_decay);

        m_out
    }
}

// Schnelle GeLU-Approximation (Hendrycks & Gimpel)
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // ≈ sqrt(2/pi)
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}
