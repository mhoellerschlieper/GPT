// main.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Entry point and CLI menu.
//           Provides:
//           - checkpoint load/save
//           - two phase training flow
//           - interactive inference and inference grid
//           - corpus pipeline (fetch/prepare/filter/pack)
// History:
//  - 2026-01-16: Restores two phase training menu flow with Enter defaults.
//  - 2026-01-17: Adds corpus menu entry for corpus pipeline execution.
//  - 2026-01-18: Fixes corpus menu compilation by using anyhow::Result and
//                adding d_val_ratio_sources to CorpusPackSpec.
// ============================================================================

#![forbid(unsafe_code)]
#![allow(warnings)]

mod augmentation;
mod corpus;
mod layers;
mod math;
mod tokenize;
mod train;
mod utils;

use crate::corpus::{Corpus, CorpusPackSpec, CorpusSourceSpec, CorpusSpec};
use crate::layers::{Embeddings, OutputProjection, TransformerBlockV2};
use crate::tokenize::Tokenizer;
use crate::train::{InferenceGrid, LLM, TrainConfig};
use crate::utils::MAX_SEQ_LEN_CANONICAL;
use anyhow::Context;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use utils::{
    Dataset, DatasetType, DROPOUT, EMBEDDING_DIM, HEADS, HIDDEN_DIM, LEARN_RATE_PRETRAIN,
    LEARN_RATE_TRAIN,
};

const S_CHECKPOINT_PATH: &str = "checkpoint.bin";
const S_TOKENIZER_VOCAB_PATH: &str = "data/tokenizer_vocab.bin";
const S_TOKENIZER_MERGES_PATH: &str = "data/tokenizer_merges.txt";

fn read_line_trimmed() -> io::Result<String> {
    let mut s_in = String::new();
    io::stdin().read_line(&mut s_in)?;
    Ok(s_in.trim().to_string())
}

fn read_usize_default(
    s_prompt: &str,
    i_default: usize,
    i_min: usize,
    i_max: usize,
) -> io::Result<usize> {
    loop {
        print!("{} [{}]: ", s_prompt, i_default);
        io::stdout().flush()?;
        let s_in = read_line_trimmed()?;
        if s_in.is_empty() {
            return Ok(i_default);
        }
        match s_in.parse::<usize>() {
            Ok(v) if v >= i_min && v <= i_max => return Ok(v),
            _ => println!(
                "Invalid input. Enter integer in range {}..={} or press Enter.",
                i_min, i_max
            ),
        }
    }
}

fn read_f32_default(s_prompt: &str, d_default: f32, d_min: f32, d_max: f32) -> io::Result<f32> {
    loop {
        print!("{} [{:.6}]: ", s_prompt, d_default);
        io::stdout().flush()?;
        let s_in = read_line_trimmed()?;
        if s_in.is_empty() {
            return Ok(d_default);
        }
        match s_in.parse::<f32>() {
            Ok(v) if v >= d_min && v <= d_max => return Ok(v),
            _ => println!(
                "Invalid input. Enter float in range {}..={} or press Enter.",
                d_min, d_max
            ),
        }
    }
}

fn read_string_default(s_prompt: &str, s_default: &str) -> io::Result<String> {
    loop {
        print!("{} [{}]: ", s_prompt, s_default);
        io::stdout().flush()?;
        let s_in = read_line_trimmed()?;
        if s_in.is_empty() {
            return Ok(s_default.to_string());
        }
        if s_in.trim().is_empty() {
            println!("Invalid input. Enter non empty string or press Enter.");
            continue;
        }
        return Ok(s_in);
    }
}

fn run_corpus_menu() -> anyhow::Result<()> {
    println!("===== CORPUS PIPELINE =====");
    println!("This builds datasets via: fetch, prepare, filter, pack");
    println!("Press Enter to accept defaults.");

    let s_root_dir = read_string_default("corpus_root_dir", "data")?;

    // Defaults must be valid http(s) URL for kind http_file.
    let s_source_id = read_string_default("source_id", "gutenberg_shakespeare_100")?;
    let s_source_url = read_string_default(
        "source_url",
        "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    )?;
    let s_source_filename = read_string_default("source_filename", "pg100.txt")?;

    let d_pretrain_ratio = read_f32_default("pretrain_ratio", 0.20, 0.0, 1.0)?;
    let d_val_ratio_sources = read_f32_default("val_ratio_sources", 0.10, 0.0, 0.5)?;

    let i_max_lines_total = read_usize_default("max_lines_total", 200000, 100, 100000000)?;
    let i_min_line_len = read_usize_default("min_line_len", 40, 1, 100000)?;
    let i_max_line_len = read_usize_default("max_line_len", 2000, 10, 200000)?;

    let s_out_pretrain_json = read_string_default("out_pretrain_json", "pretraining_data_de.json")?;
    let s_out_main_json = read_string_default("out_main_json", "chat_training_data_de.json")?;

    let corpus = Corpus::new(s_root_dir);

    let spec = CorpusSpec {
        v_sources: vec![CorpusSourceSpec {
            s_id: s_source_id,
            s_kind: "http_file".to_string(),
            s_url: s_source_url,
            s_filename: s_source_filename,
            m_params: HashMap::new(),
        }],
        pack: CorpusPackSpec {
            s_out_pretrain_json,
            s_out_main_json,
            d_pretrain_ratio,
            i_max_lines_total,
            i_min_line_len,
            i_max_line_len,
            d_val_ratio_sources,
        },
    };

    println!("CORPUS: fetch");
    let manifest = corpus.corpus_fetch(&spec).context("corpus_fetch failed")?;

    println!("CORPUS: prepare");
    corpus
        .corpus_prepare(&spec, &manifest)
        .context("corpus_prepare failed")?;

    println!("CORPUS: filter");
    corpus.corpus_filter(&spec).context("corpus_filter failed")?;

    println!("CORPUS: pack");
    corpus.corpus_pack(&spec).context("corpus_pack failed")?;

    println!("CORPUS: done");
    Ok(())
}

fn prepare_tokenizer_from_dataset(dataset: &Dataset, i_merge_limit: usize) -> Tokenizer {
    if Path::new(S_TOKENIZER_VOCAB_PATH).exists() && Path::new(S_TOKENIZER_MERGES_PATH).exists() {
        match Tokenizer::load(S_TOKENIZER_VOCAB_PATH, S_TOKENIZER_MERGES_PATH) {
            Ok(tok) => {
                println!("Tokenizer loaded from persistence.");
                return tok;
            }
            Err(e) => {
                eprintln!("Warning: tokenizer load failed: {}", e);
            }
        }
    }

    println!("Training tokenizer BPE with merge_limit={} ...", i_merge_limit);

    let mut v_corpus: Vec<String> = Vec::new();
    v_corpus.extend(dataset.pretraining_data.iter().cloned());
    v_corpus.extend(dataset.chat_training_data.iter().cloned());

    let mut tok = Tokenizer::new_byte_bpe_reversible();
    tok.train_bpe(&v_corpus, i_merge_limit);

    match tok.save(S_TOKENIZER_VOCAB_PATH, S_TOKENIZER_MERGES_PATH) {
        Ok(_) => println!(
            "Tokenizer saved to {}, {}",
            S_TOKENIZER_VOCAB_PATH, S_TOKENIZER_MERGES_PATH
        ),
        Err(e) => eprintln!("Warning: tokenizer save failed: {}", e),
    }

    tok
}

fn build_llm(tokenizer: Tokenizer, i_max_seq_len: usize) -> LLM {
    let i_vocab_size = tokenizer.vocab_size();

    let embeddings = Embeddings::from_tokenizer(&tokenizer);

    let block_1 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let block_2 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let block_3 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let block_4 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let block_5 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let block_6 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);

    let output_projection = OutputProjection::new(EMBEDDING_DIM, i_vocab_size);

    LLM::new(
        tokenizer,
        vec![
            Box::new(embeddings),
            Box::new(block_1),
            Box::new(block_2),
            Box::new(block_3),
            Box::new(block_4),
            Box::new(block_5),
            Box::new(block_6),
            Box::new(output_projection),
        ],
        i_max_seq_len,
    )
}

fn default_cfg_pretrain() -> TrainConfig {
    let mut cfg = TrainConfig::default_for_debug();
    cfg.i_epochs = 10;
    cfg.i_steps_per_epoch = 500;
    cfg.i_batch_size = 16;
    cfg.d_val_ratio = 0.10;
    cfg.d_lr = LEARN_RATE_PRETRAIN;
    cfg.i_max_seq_len = 256;
    cfg.i_min_window_len = 32;

    // Pretraining should typically not use assistant-only masking.
    cfg.b_loss_mask_assistant_only = false;

    cfg
}

fn default_cfg_main() -> TrainConfig {
    let mut cfg = TrainConfig::default_for_debug();
    cfg.i_epochs = 100;
    cfg.i_steps_per_epoch = 300;
    cfg.i_batch_size = 16;
    cfg.d_val_ratio = 0.10;
    cfg.d_lr = LEARN_RATE_TRAIN;
    cfg.i_max_seq_len = 256;
    cfg.i_min_window_len = 32;

    // Main training (chat) can use assistant-only masking.
    cfg.b_loss_mask_assistant_only = true;

    cfg
}

fn menu_cfg_with_defaults(s_title: &str, cfg_in: &TrainConfig) -> io::Result<TrainConfig> {
    println!("{}", s_title);
    println!("Press Enter to accept defaults.");

    let mut cfg = cfg_in.clone();

    cfg.i_epochs = read_usize_default("epochs", cfg.i_epochs, 1, 1000)?;
    cfg.i_steps_per_epoch = read_usize_default("steps_per_epoch", cfg.i_steps_per_epoch, 10, 100000)?;
    cfg.i_batch_size = read_usize_default("batch_size", cfg.i_batch_size, 1, 1024)?;
    cfg.d_val_ratio = read_f32_default("val_ratio", cfg.d_val_ratio, 0.0, 0.5)?;
    cfg.d_lr = read_f32_default("learn_rate", cfg.d_lr, 1e-6, 1.0)?;
    cfg.i_max_seq_len = read_usize_default("max_seq_len", cfg.i_max_seq_len, 32, MAX_SEQ_LEN_CANONICAL)?;
    cfg.i_min_window_len = read_usize_default("min_window_len", cfg.i_min_window_len, 8, MAX_SEQ_LEN_CANONICAL)?;

    if cfg.i_min_window_len > cfg.i_max_seq_len {
        cfg.i_min_window_len = cfg.i_max_seq_len;
    }

    Ok(cfg)
}

fn run_menu(mut llm: LLM, dataset: Dataset) -> io::Result<()> {
    loop {
        println!();
        println!("===== MAIN MENU =====");
        println!("  l - load checkpoint");
        println!("  s - save checkpoint");
        println!("  t - train (pretraining + main training)");
        println!("  b - ask model (interactive)");
        println!("  g - inference grid on fixed prompts");
        println!("  c - corpus pipeline (fetch + prepare + filter + pack)");
        println!("  e - exit");
        print!("choice: ");
        io::stdout().flush()?;

        let s_choice = read_line_trimmed()?.to_lowercase();

        match s_choice.as_str() {
            "l" => match llm.load_checkpoint(S_CHECKPOINT_PATH) {
                Ok(_) => println!("Checkpoint loaded."),
                Err(e) => println!("Load failed: {}", e),
            },
            "s" => match llm.save_checkpoint(S_CHECKPOINT_PATH) {
                Ok(_) => println!("Checkpoint saved."),
                Err(e) => println!("Save failed: {}", e),
            },
            "t" => {
                let cfg_pre = menu_cfg_with_defaults("PHASE A: pretraining config", &default_cfg_pretrain())?;
                let cfg_main = menu_cfg_with_defaults("PHASE B: main training config", &default_cfg_main())?;

                let v_pretrain = dataset.pretraining_data.clone();
                let v_main = dataset.chat_training_data.clone();

                println!(
                    "Training start: pretrain_samples={} main_samples={}",
                    v_pretrain.len(),
                    v_main.len()
                );

                llm.train_two_phase(&v_pretrain, &v_main, &cfg_pre, &cfg_main);

                println!("Training finished.");
            }
            "b" => {
                println!("Interactive mode. Type 'done' to exit.");
                loop {
                    print!("user> ");
                    io::stdout().flush()?;
                    let s_q = read_line_trimmed()?;
                    if s_q.eq_ignore_ascii_case("done") {
                        break;
                    }
                    if s_q.is_empty() {
                        continue;
                    }
                    let s_prompt = format!("User: {} Assistant: ", s_q);
                    print!("assistant> ");
                    io::stdout().flush()?;
                    let _ = llm.predict(&s_prompt);
                }
            }
            "g" => {
                let grid = InferenceGrid::conservative_default();
                let v_prompts: Vec<String> = vec![
                    "User: Erklaere kurz den Unterschied zwischen TCP und UDP. Assistant: ".to_string(),
                    "User: Schreibe eine kurze E Mail zur Terminbestaetigung. Assistant: ".to_string(),
                    "User: Nenne drei Vorteile von Rust gegenueber C plus plus. Assistant: ".to_string(),
                ];
                llm.run_inference_grid(&v_prompts, &grid);
            }
            "c" => match run_corpus_menu() {
                Ok(_) => println!("Corpus pipeline finished."),
                Err(e) => println!("Corpus pipeline failed: {}", e),
            },
            "e" => {
                println!("Exiting.");
                break;
            }
            _ => println!("Unknown choice."),
        }
    }

    Ok(())
}

fn main() {
    println!("Initialization running ...");

    let dataset = Dataset::new(
        "data/pretraining_data_de.json".into(),
        "data/chat_training_data_de.json".into(),
        DatasetType::JSON,
    );

    let tokenizer = prepare_tokenizer_from_dataset(&dataset, 50_000);

    let i_max_seq_len_runtime: usize = MAX_SEQ_LEN_CANONICAL;
    let mut llm = build_llm(tokenizer, i_max_seq_len_runtime);

    println!("=== MODEL INFO ===");
    println!("Network: {}", llm.network_description());
    println!("Total parameters: {}", llm.total_parameters());
    println!("Max seq len: {}", i_max_seq_len_runtime);

    let _ = llm.load_checkpoint(S_CHECKPOINT_PATH);

    if let Err(e) = run_menu(llm, dataset) {
        eprintln!("Fatal error: {}", e);
    }
}
