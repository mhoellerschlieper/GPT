// ===========================================
// Autor   : Marcus Schlieper (ExpChat_ai)
// Erstellt: 22_11_2025
// Datei   : utils.rs – Hilfsfunktionen
// Historie: 22_11_2025  MS  Sliding-Window
// ===========================================
use crate::MAX_SEQ_LEN;

/// Zerteilt eine Sequenz in überlappende Chunks.
/// Parameter
/// * `tokens`      – Eingabesequenz (Token-IDs).
/// * `i_overlap`   – Anzahl überlappender Tokens zwischen zwei Chunks.
/// Rückgabe
/// * Vektor von Sequenzen, jede höchstens MAX_SEQ_LEN lang.
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
