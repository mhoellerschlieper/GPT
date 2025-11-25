/***********************************************************************
*  layer_time2vec.rs  –  kontinuierliche Zeit-Einbettung (Kazemi & Mehri, 2019)
***********************************************************************/
use ndarray::{Array1, Array2, Axis};

#[derive(Clone)]
pub struct Time2Vec {
    /// linearer Anteil   τ·w₀ + b₀
    w_lin:  f32,
    b_lin:  f32,
    /// periodische Anteile sin(τ·wᵢ + bᵢ)
    v_freq: Array1<f32>,
    v_bias: Array1<f32>,
}

impl Time2Vec {
    pub fn new(i_embedding_dim: usize) -> Self {
        let i_periodic = i_embedding_dim - 1; // 1 linear + Rest sinus
        Time2Vec {
            w_lin:  0.01,
            b_lin:  0.0,
            v_freq: Array1::ones(i_periodic) * 0.01,
            v_bias: Array1::zeros(i_periodic),
        }
    }

    /// τ = Zeitstempel in Sekunden (oder jeder beliebigen kontinuierlichen Einheit)
    pub fn encode(&self, d_timestamp: f32) -> Array1<f32> {
        let mut v_out = Array1::<f32>::zeros(self.v_freq.len() + 1);
        // linearer Kanal
        v_out[0] = d_timestamp * self.w_lin + self.b_lin;
        // periodische Kanäle
        for (idx, (&w, &b)) in self.v_freq.iter().zip(self.v_bias.iter()).enumerate() {
            v_out[idx + 1] = (d_timestamp * w + b).sin();
        }
        v_out
    }

    /// erweitert ein Batchelement um Zeitfeatures
    pub fn add_time_axis(&self, m_tokens: &Array2<f32>, d_timestamp: f32) -> Array2<f32> {
        let v_time = self.encode(d_timestamp);
        let i_seq   = m_tokens.dim().0;
        let i_embed = m_tokens.dim().1;

        // Broadcast: Sequenzlänge × Embedding
        let mut m_out = m_tokens.clone();
        for i_row in 0..i_seq {
            for j in 0..i_embed {
                m_out[[i_row, j]] += v_time[j];
            }
        }
        m_out
    }
}