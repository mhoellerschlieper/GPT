// main.rs
// ============================================================================
// Autor:    Marcus Schlieper (ExpChat.ai)
// Kontakt:  mschlieper@ylook.de | Tel: 49 2338 8748862 | Mobil: 49 15115751864
// Firma:    ExpChat.ai – Der KI Chat Client fuer den Mittelstand
// Adresse:  Epscheider Str21, 58339 Breckerfeld
// Hinweis:  Einstiegspunkt. Initialisiert Tokenizer, Modell, Menu und Training.
// ============================================================================

#![forbid(unsafe_code)]

mod layers;
mod train;
mod math;
mod utils;

use std::io::{self, Write};
use std::path::Path;

use layers::{Embeddings, OutputProjection, TransformerBlockV2};
use train::LLM;
use utils::{Dataset, DatasetType, Tokenizer, EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT, MAX_SEQ_LEN, LEARN_RATE_PRETRAIN, LEARN_RATE_TRAIN};

fn prepare_tokenizer(corpus: &[String], i_merge_limit: usize) -> Tokenizer {
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

    println!("Starte einmaliges BPE-Training ({i_merge_limit} Merges) ...");
    let mut tokenizer = Tokenizer::new_byte_level();
    tokenizer.train_bpe(corpus, i_merge_limit);
    tokenizer.save(P_VOCAB, P_MERGES).expect("Tokenizer-Persistenz fehlgeschlagen");
    println!("Tokenizer gespeichert unter {P_VOCAB}, {P_MERGES}");
    tokenizer
}

fn run_menu(llm: &mut LLM, dataset: &Dataset) -> io::Result<()> {
    loop {
        println!("\n===== HAUPTMENU =====");
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
                let i_epochs_pretrain: usize = loop {
                    print!("Wie viele Epochen Vortraining? ");
                    std::io::stdout().flush()?;
                    let mut s_input = String::new();
                    std::io::stdin().read_line(&mut s_input)?;
                    match s_input.trim().parse::<usize>() {
                        Ok(val) if val > 0 => break val,
                        _ => println!("Bitte eine positive Ganzzahl eingeben."),
                    }
                };

                let i_epochs_train: usize = loop {
                    print!("Wie viele Epochen Haupttraining? ");
                    std::io::stdout().flush()?;
                    let mut s_input = String::new();
                    std::io::stdin().read_line(&mut s_input)?;
                    match s_input.trim().parse::<usize>() {
                        Ok(val) if val > 0 => break val,
                        _ => println!("Bitte eine positive Ganzzahl eingeben."),
                    }
                };

                let i_batch_size: usize = loop {
                    print!("Batch-Groesse (z. B. 16): ");
                    std::io::stdout().flush()?;
                    let mut s_input = String::new();
                    std::io::stdin().read_line(&mut s_input)?;
                    match s_input.trim().parse::<usize>() {
                        Ok(val) if val > 0 => break val,
                        _ => println!("Bitte eine positive Ganzzahl eingeben."),
                    }
                };

                let pretraining_examples: Vec<&str> = dataset.pretraining_data.iter().map(|s| s.as_str()).collect();
                let chat_training_examples: Vec<&str> = dataset.chat_training_data.iter().map(|s| s.as_str()).collect();

                println!("Starte Pre-Training ({i_epochs_pretrain} Epochen, Batch={i_batch_size}) ...");
                llm.train(pretraining_examples, i_epochs_pretrain, LEARN_RATE_PRETRAIN, i_batch_size);

                println!("Starte Instruction-Tuning ({i_epochs_train} Epochen, Batch={i_batch_size}) ...");
                llm.train(chat_training_examples, i_epochs_train, LEARN_RATE_TRAIN, i_batch_size);

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

fn main() {
    println!("Initialisierung laeuft ...");

    let _ = rayon::ThreadPoolBuilder::new().num_threads(8).build_global();

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
    let transformer_block_5 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_6 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_7 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_8 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_9 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_10 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_11 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);
    let transformer_block_12 = TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, HEADS, DROPOUT);

    let output_projection = OutputProjection::new(EMBEDDING_DIM, i_vocab_size);

    let mut llm = LLM::new(
        tokenizer,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(transformer_block_4),
            Box::new(transformer_block_5),
            Box::new(transformer_block_6),
            Box::new(transformer_block_7),
            Box::new(transformer_block_8),
            Box::new(transformer_block_9),
            Box::new(transformer_block_10),
            Box::new(transformer_block_11),
            Box::new(transformer_block_12),
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

    if let Err(e) = run_menu(&mut llm, &dataset) {
        eprintln!("Fehler im Menu: {e}");
    }
}
