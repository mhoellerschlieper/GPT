// ===========================================
// Autor   : Marcus Schlieper (ExpChat.ai)
// Datei   : utils.rs – Hilfsfunktionen
// Historie: 22_11_2025  MS  Sliding-Window
//           26_11_2025  MS  Typen/Droput-Fixes
// ===========================================
use ndarray::Array2;
use rand::Rng;

use half::{f16, bf16};

pub const MAX_SEQ_LEN: usize = 80;

/// Zerteilt eine Sequenz in ueberlappende Chunks.
pub fn chunk_sequence(v_tokens: &[usize], i_overlap: usize) -> Vec<Vec<usize>> {
    assert!(i_overlap < MAX_SEQ_LEN, "overlap must be smaller than MAX_SEQ_LEN");

    let mut v_chunks: Vec<Vec<usize>> = Vec::new();
    let mut i_start: usize = 0;

    while i_start < v_tokens.len() {
        let i_end: usize = usize::min(i_start + MAX_SEQ_LEN, v_tokens.len());
        v_chunks.push(v_tokens[i_start..i_end].to_vec());

        if i_end == v_tokens.len() {
            break;
        }
        i_start = i_end.saturating_sub(i_overlap);
    }
    v_chunks
}

/// Bernoulli-Dropout (inverted).
pub fn dropout_inplace(m_tensor: &mut Array2<f32>, f_rate: f32) {
    if f_rate <= 0.0 {
        return;
    }
    let p_drop = f_rate.clamp(0.0, 1.0);
    let scale = if p_drop < 1.0 { 1.0 / (1.0 - p_drop) } else { 0.0 };

    let mut rng = rand::thread_rng();
    for elem in m_tensor.iter_mut() {
        let drop_now: bool = rng.r#gen::<f32>() < p_drop;
        if drop_now {
            *elem = 0.0;
        } else {
            *elem *= scale;
        }
    }
}

/// Elementweise L2-Gewichtsnormierung
pub fn apply_weight_decay(m_tensor: &mut Array2<f32>, f_lambda: f32) {
    if f_lambda > 0.0 {
        let alpha = 1.0 - f_lambda;
        m_tensor.mapv_inplace(|v| v * alpha);
    }
}

pub fn to_f16(m: &Array2<f32>) -> Array2<f16> {
    m.mapv(f16::from_f32)
}
pub fn from_f16(m: &Array2<f16>) -> Array2<f32> {
    m.mapv(|h| f32::from(h))
}
pub fn to_bf16(m: &Array2<f32>) -> Array2<bf16> {
    m.mapv(bf16::from_f32)
}
pub fn from_bf16(m: &Array2<bf16>) -> Array2<f32> {
    m.mapv(|h| f32::from(h))
}
