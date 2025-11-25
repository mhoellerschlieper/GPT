// ===========================================
// Autor   : Marcus Schlieper (ExpChat_ai)
// Erstellt: 22_11_2025
// Datei   : utils.rs – Hilfsfunktionen
// Historie: 22_11_2025  MS  Sliding-Window
// ===========================================
use crate::MAX_SEQ_LEN;
use ndarray::{Array2, ArrayBase, DataMut, Ix2};
use rand::{Rng, rng};

/// Zerteilt eine Sequenz in überlappende Chunks.
/// * tokens: Eingabesequenz (Token-IDs)
/// * i_overlap: Anzahl überlappender Tokens
/// Rückgabe: Vektor von Sequenzen, jede höchstens MAX_SEQ_LEN lang.
pub fn chunk_sequence(tokens: &[usize], i_overlap: usize) -> Vec<Vec<usize>> {
    assert!(i_overlap < MAX_SEQ_LEN, "overlap must be smaller than MAX_SEQ_LEN");

    let mut v_chunks: Vec<Vec<usize>> = Vec::new();
    let mut i_start: usize = 0;

    while i_start < tokens.len() {
        let i_end: usize = usize::min(i_start + MAX_SEQ_LEN, tokens.len());
        v_chunks.push(tokens[i_start..i_end].to_vec());

        if i_end == tokens.len() {
            break;
        }
        i_start = i_end.saturating_sub(i_overlap);
    }
    v_chunks
}

/***********************************************************************
*  Hilfsfunktionen (Drop-out, Gewichtsnormierung)
***********************************************************************/

/// Wendet Bernoulli-Dropout an (in-place).
/// f_rate ist die Dropout-Rate (z. B. 0.1 = 10%).
/// Nutzt „inverted dropout“ (Skalierung), damit der Erwartungswert stabil bleibt.
pub fn dropout_inplace<A>(m_tensor: &mut ArrayBase<A, Ix2>, f_rate: f32)
where
    A: DataMut<Elem = f32>,
{
    if f_rate <= 0.0 {
        return;
    }
    let p_drop = f_rate.clamp(0.0, 1.0);
    let scale = if p_drop < 1.0 { 1.0 / (1.0 - p_drop) } else { 0.0 };

    let mut r = rng();
    for elem in m_tensor.iter_mut() {
        let drop_now: bool = r.random::<f32>() < p_drop;
        if drop_now {
            *elem = 0.0;
        } else {
            *elem *= scale;
        }
    }
}

/// Elementweise L2-Gewichtsnormierung („Weight-Decay light“)
pub fn apply_weight_decay(m_tensor: &mut Array2<f32>, f_lambda: f32) {
    if f_lambda > 0.0 {
        let alpha = 1.0 - f_lambda;
        // In-place Skalierung (vermeidet Borrow-Konflikte)
        m_tensor.mapv_inplace(|v| v * alpha);
        // Alternativ (falls verfügbar):
        // *m_tensor *= alpha;
    }
}
