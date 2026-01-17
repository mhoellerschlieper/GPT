// utils.rs
// ============================================================================
// Autor:   Marcus Schlieper (ExpChat.ai)
// Hinweis: Config, Tokenizer, Dataset-Lader, Sequenz-Chunking.
// ============================================================================

#![forbid(unsafe_code)]

use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

// ---------------- Config ----------------

pub const MAX_SEQ_LEN_CANONICAL: usize = 256;

// Backward compatible alias: use only if other modules still reference MAX_SEQ_LEN.
pub const MAX_SEQ_LEN: usize = MAX_SEQ_LEN_CANONICAL;

pub const EMBEDDING_DIM: usize = 256;
pub const HIDDEN_DIM: usize = 1024;
pub const HEADS: usize = 8;
pub const DROPOUT: f32 = 0.1;

pub const LEARN_RATE_PRETRAIN: f32 = 1e-3;
pub const LEARN_RATE_TRAIN: f32 = 5e-4;

pub const S_EOS: &str = "</s>";
pub const S_UNK: &str = "<unk>";

// ---------------- Dataset ----------------

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(pretraining_data_path: String, chat_training_data_path: String, type_of_data: DatasetType) -> Self {
        let (pretraining_data, chat_training_data) = match type_of_data {
            DatasetType::CSV => (get_data_from_csv(pretraining_data_path), get_data_from_csv(chat_training_data_path)),
            DatasetType::JSON => (get_data_from_json(pretraining_data_path), get_data_from_json(chat_training_data_path)),
        };
        Dataset { pretraining_data, chat_training_data }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    let data_json = fs::read_to_string(path).expect("Failed to read data file");
    let data: Vec<String> = serde_json::from_str(&data_json).expect("Failed to parse data file");
    data
}

pub fn get_data_from_csv(path: String) -> Vec<String> {
    let file_path = Path::new(&path);
    let file = fs::File::open(file_path)
        .unwrap_or_else(|e| panic!("CSV-Datei {:?} konnte nicht geoeffnet werden: {}", file_path, e));

    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut v_data: Vec<String> = Vec::new();

    for (_i, result) in rdr.records().enumerate() {
        let record = result.expect("Fehler beim Lesen eines CSV-Datensatzes");
        let mut s_line: String = record.iter().collect::<Vec<&str>>().join(",");
        s_line = s_line.trim_end().to_string();
        if !s_line.ends_with(S_EOS) {
            s_line.push_str(S_EOS);
        }
        v_data.push(s_line);
    }

    v_data
}

