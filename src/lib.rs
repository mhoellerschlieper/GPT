// =============================================
// lib.rs
// =============================================
pub mod utils;
pub mod adam;
pub mod dataset_loader;
pub mod embeddings;

pub mod feed_forward;
pub mod feed_forward_geglu;

pub mod layer_norm;
pub mod llm;

pub mod layer_output_projection;
pub mod layer_self_attention;
pub mod layer_pos_encoding;
pub mod layer_time2vec;

pub mod transformer;
pub mod transformer_block_v2;
pub mod multi_head_attention;
pub mod tokenizer_bpe;


// Re-export key structs for easier access
pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use llm::{LLM, Layer};
    
// Constants
pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
