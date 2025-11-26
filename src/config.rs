// src/config.rs
// Zentrale Hyperparameter (ein Ort für alle Konstanten)
pub const MAX_SEQ_LEN: usize   = 512;   // Sequenz-Länge (an dein Training anpassen)
pub const EMBEDDING_DIM: usize = 256;   // Dimension der Embeddings
pub const HIDDEN_DIM: usize    = 1024;  // Hidden-Dimension im Feed-Forward