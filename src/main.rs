// ============================================================================
// Autor      : Marcus Schlieper (ExpChat.ai)
// Erstellt   : 22.11.2025
// Datei      : main.rs – Einstiegspunkt der Anwendung
// Historie   : 22.11.2025  MS  Erste Version mit interaktivem Menü
//              22.11.2025  MS  BPE-Tokenizer-Training + Persistenz integriert
//              26.11.2025  MS  Build-Fixes, Modulpfade, Typen, Menu-Parsing
// ============================================================================

use std::{
    io::{self, Write},
    path::Path,
};

use crate::dataset_loader::{Dataset, DatasetType};

use crate::{
    embeddings::Embeddings,
    layer_output_projection::OutputProjection,
    llm::LLM,
    transformer_block_v2::TransformerBlockV2,
};

mod config;
mod adam;
mod dataset_loader;
mod embeddings;
mod feed_forward;
mod layer_norm;
mod llm;
mod layer_output_projection;
mod layer_self_attention;
mod tokenizer_bpe;
mod transformer_block_v2;

// neue Module (Fixes)
mod multi_head_attention;
mod feed_forward_geglu;
mod layer_time2vec;
mod utils;
mod layer_pos_encoding;

use crate::tokenizer_bpe::Tokenizer;
use crate::config::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};

const LEARN_RATE_PRETRAIN: f32 = 5e-4;
const LEARN_RATE_TRAIN: f32 = 1e-4;

// Hyperparameter fuer TransformerBlockV2
const HEADS: usize = 8;
const DROPOUT: f32 = 0.1;

/// ---------------------------------------------------------------------------
/// prepare_tokenizer
/// ---------------------------------------------------------------------------
fn prepare_tokenizer(korpus: &[String], i_merge_limit: usize) -> Tokenizer {
    const P_VOCAB: &str = "data/bpe_vocab.bin";
    const P_MERGES: &str = "data/bpe_merges.txt";

    if Path::new(P_VOCAB).exists() && Path::new(P_MERGES).exists() {
        match Tokenizer::load(P_VOCAB, P_MERGES) {
            Ok(tok) => {
                println!("Tokenizer aus Persistenz geladen.");
                return tok;
            }
            Err(e) => eprintln!("Warnung: Tokenizer konnte nicht geladen werden: {e}"),
        }
    }

    println!("Starte einmaliges BPE-Training ({} Merges) …", i_merge_limit);
    let mut tokenizer = Tokenizer::new_byte_level();
    tokenizer.train_bpe(korpus, i_merge_limit);

    tokenizer
        .save(P_VOCAB, P_MERGES)
        .expect("Tokenizer-Persistenz fehlgeschlagen");
    println!("Tokenizer gespeichert unter {}, {}", P_VOCAB, P_MERGES);

    tokenizer
}

/// ---------------------------------------------------------------------------
/// run_menu
/// ---------------------------------------------------------------------------
fn run_menu(llm: &mut LLM, dataset: &Dataset) -> io::Result<()> {
    loop {
        println!("\n===== HAUPTMENÜ =====");
        println!("  l – Modell laden");
        println!("  s – Modell speichern");
        println!("  t – Modell trainieren");
        println!("  b – Frage an das Modell stellen");
        println!("  e – Programm beenden");
        print!("Ihre Wahl: ");
        io::stdout().flush()?;

        let mut s_choice = String::new();
        io::stdin().read_line(&mut s_choice)?;
        match s_choice.trim().to_lowercase().as_str() {
            "l" => match llm.load_checkpoint("checkpoint.bin") {
                Ok(_) => println!("Checkpoint erfolgreich geladen."),
                Err(e) => println!("Fehler beim Laden: {e}"),
            },
            "s" => match llm.save_checkpoint("checkpoint.bin") {
                Ok(_) => println!("Checkpoint erfolgreich gespeichert."),
                Err(e) => println!("Fehler beim Speichern: {e}"),
            },
            "t" => {
                let i_epochs: usize = loop {
                    print!("Wie viele Epochen sollen verwendet werden? ");
                    io::stdout().flush()?;
                    let mut s_input = String::new();
                    io::stdin().read_line(&mut s_input)?;
                    match s_input.trim().parse::<usize>() {
                        Ok(val) if val > 0 => break val,
                        _ => println!("Bitte eine positive Ganzzahl eingeben."),
                    }
                };

                let pretraining_examples: Vec<&str> = dataset
                    .pretraining_data
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let chat_training_examples: Vec<&str> = dataset
                    .chat_training_data
                    .iter()
                    .map(|s| s.as_str())
                    .collect();

                println!("Starte Pre-Training ({} Epochen) …", i_epochs);
                llm.train(pretraining_examples, i_epochs, LEARN_RATE_PRETRAIN);

                println!("Starte Instruction-Tuning ({} Epochen) …", i_epochs);
                llm.train(chat_training_examples, i_epochs, LEARN_RATE_TRAIN);

                println!("Training abgeschlossen.");
            }
            "b" => {
                println!("Eingabemodus ('fertig' zum Beenden):");
                loop {
                    print!("Frage: ");
                    io::stdout().flush()?;
                    let mut s_query = String::new();
                    io::stdin().read_line(&mut s_query)?;
                    let s_trimmed = s_query.trim();
                    if s_trimmed.eq_ignore_ascii_case("fertig") {
                        break;
                    }
                    let s_formatted = format!("User: {}", s_trimmed);
                    let s_answer = llm.predict(&s_formatted);
                    println!("Antwort: {s_answer}");
                }
            }
            "e" => {
                println!("Programm wird beendet.");
                break;
            }
            _ => println!("Unbekannte Auswahl – bitte erneut versuchen."),
        }
    }
    Ok(())
}

/// ---------------------------------------------------------------------------
/// main
/// ---------------------------------------------------------------------------
fn main() {
    println!("Initialisierung läuft …");

    let dataset = Dataset::new(
        "data/pretraining_data_de.json".into(),
        "data/chat_training_data_de.json".into(),
        DatasetType::JSON,
    );

    let mut korpus: Vec<String> =
        Vec::with_capacity(dataset.pretraining_data.len() + dataset.chat_training_data.len());
    korpus.extend(dataset.pretraining_data.clone());
    korpus.extend(dataset.chat_training_data.clone());

    let tokenizer = prepare_tokenizer(&korpus, 50_000);
    let i_vocab_size = tokenizer.vocab_size();

    let embeddings = Embeddings::from_tokenizer(&tokenizer);

    let transformer_block_1 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_2 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_3 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_4 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);

    let output_projection = OutputProjection::new(EMBEDDING_DIM, i_vocab_size);

    let mut llm = LLM::new(
        tokenizer,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(transformer_block_4),
            Box::new(output_projection),
        ],
    );

    let _ = llm.load_checkpoint("checkpoint.bin");

    println!("=== MODELL-INFO ===");
    println!("Netzwerkarchitektur  : {}", llm.network_description());
    println!(
        "Konfiguration         : max_seq_len={}, embedding_dim={}, hidden_dim={}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Gesamtparameter       : {}", llm.total_parameters());

    let s_sample_input = "User: Wie entstehen Gebirge?";
    println!("\nBeispiel-Input  : {s_sample_input}");
    let s_answer = llm.predict(s_sample_input);
    println!("Vorhersage (raw): {}", s_answer);

    if let Err(e) = run_menu(&mut llm, &dataset) {
        eprintln!("Fehler im Menü: {e}");
    }
}
