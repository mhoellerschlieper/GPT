// ===========================================================================
//  Datei:        llm.rs
//  Projekt:      Lightweight-Language-Model (LLM)
//  Modul:        Kernlogik
// ---------------------------------------------------------------------------
//  Zweck
//  -----
//  Das vorliegende Modul implementiert die zentrale Ablaufsteuerung eines
//  kompakten, vollständig differentiierbaren Sprachmodelles.  Der Fokus liegt
//  auf einem speicherökonomischen Layer-Stack, der neben klassischen
//  Embedding-, Transformer- und Projektionsebenen eine Sliding-Window-Logik
//  zur sequentiellen Verarbeitung überlanger Eingaben (> MAX_SEQ_LEN)
//  umfasst.  Darüber hinaus stellt das Modul Funktionalitäten zum Training,
//  zur Inferenz (autoregessive Vorhersage) und zur Persistenz (Check-pointing)
//  bereit.
//
//  Strukturüberblick
//  -----------------
//      ┌──────────────────────────────────────────────────────────────┐
//      │ Tokenizer        – Byte-Level-BPE, integriert in LLM        │
//      ├──────────────────────────────────────────────────────────────┤
//      │ Layer-Stack      – Embeddings ▸ Transformer ▸ Projection    │
//      ├──────────────────────────────────────────────────────────────┤
//      │ Sliding-Window   – Chunking mit 20 % Überlappung            │
//      ├──────────────────────────────────────────────────────────────┤
//      │ Training          Soft-max + Cross-Entropy + Adam           │
//      ├──────────────────────────────────────────────────────────────┤
//      │ Check-pointing    Serialisierung mittels bincode            │
//      └──────────────────────────────────────────────────────────────┘
//
//  Autor:        Marcus Schlieper
//  Organisation: ExpChat.ai – Der KI-Chat-Client für den Mittelstand
//  Kontakt:      Tel.  +49 2338 8748862 | Mobil +49 151 1575 1864
//  E-Mail:       mschlieper@ylook.de
//  Historie:
//      2025-11-22   Initiale Neufassung
//      2025-11-22   Integration der Sliding-Window-Logik
// ---------------------------------------------------------------------------
//  Sicherheitshinweis
//  ------------------
//  Runtime-Fehler (u. a. fehlerhafte Tensor-Formate, fehlende Checkpoints)
//  werden mittels expliziter Assertions bzw. anyhow-Kontextmeldungen
//  abgefangen.  Diese Vorgehensweise erleichtert die Fehlersuche während der
//  Entwicklungsphase, sollte jedoch in produktiven Umgebungen um feinere
//  Fehlerbehandlungen erweitert werden.
// ===========================================================================
use crate::config::EMBEDDING_DIM;
use crate::config::HIDDEN_DIM;
use crate::config::MAX_SEQ_LEN;
use crate::tokenizer_bpe::Tokenizer;
use crate::transformer_block_v2::TransformerBlockV2;
use crate::{embeddings::Embeddings, layer_output_projection::OutputProjection};
use anyhow::{Context, Result};
use bincode::{config, decode_from_std_read, encode_into_std_write};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ndarray::Array2;
use ndarray::Axis;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{Duration, Instant};

use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::{any::Any, io::ErrorKind};

// ----------------------------- Layer-Trait ----------------------------------
/**
 *  Definiert die minimale Schnittstelle, die jede Schicht implementieren
 *  muss, um in den generischen Layer-Stack aufgenommen werden zu können.
 *
 *  Hinweis: `as_any(_mut)` ermöglicht Down-Casting bei der (De-)Serialisierung.
 */
pub trait Layer {
    fn layer_type(&self) -> &str;
    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32>;
    fn backward(&mut self, grads: &ndarray::Array2<f32>, d_lr: f32) -> ndarray::Array2<f32>;
    fn parameters(&self) -> usize;

    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// ------------------------------- LLM ----------------------------------------
/**
 *  Datenstruktur des gesamten Modells
 *
 *  Felder
 *  ------
 *  tokenizer : Tokenizer  – byte-basiertes BPE-Vokabular
 *  network   : Vec<Box<dyn Layer>>
 *                Sequenzieller Layer-Stack; die letzte Schicht MUSS eine
 *                `OutputProjection`-Instanz sein, da nur diese eine
 *                Projektion in den Vokabularraum vornimmt.
 */
#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub tokenizer: Tokenizer,
    pub network: Vec<Box<dyn Layer>>,
}

// ------------------------- Hilfsfunktionen ----------------------------------
/**
 *  Zerteilt eine Token-Sequenz in sich überlappende Fenster
 *  (Greedy-Strategy).  Diese Vorgehensweise erlaubt eine annähernd
 *  kontextkonsistente Verarbeitung beliebig langer Texte.
 *
 *  Parameter
 *  ---------
 *  v_tokens  : Eingabesequenz
 *  i_overlap : Anzahl Tokens, die zwischen zwei Fenstern überlappen
 *
 *  Rückgabe
 *  --------
 *  `Vec<Vec<usize>>` – Liste von Chunks, jeweils ≤ MAX_SEQ_LEN
 */
fn chunk_sequence(v_tokens: &[usize], i_overlap: usize) -> Vec<Vec<usize>> {
    assert!(
        i_overlap < MAX_SEQ_LEN,
        "overlap must be smaller than MAX_SEQ_LEN"
    );
    let mut v_chunks: Vec<Vec<usize>> = Vec::new();
    let mut i_start: usize = 0;

    while i_start < v_tokens.len() {
        let i_end = usize::min(i_start + MAX_SEQ_LEN, v_tokens.len());
        v_chunks.push(v_tokens[i_start..i_end].to_vec());
        if i_end == v_tokens.len() {
            break;
        }
        i_start = i_end.saturating_sub(i_overlap);
    }
    v_chunks
}

// ------------------------------- Implementierung ----------------------------
impl LLM {
    // ------------------------------------------------------------------------
    // new
    // ------------------------------------------------------------------------
    /// Erstellt ein LLM-Objekt mit geprüftem Layer-Stack
    pub fn new(tokenizer: Tokenizer, layers: Vec<Box<dyn Layer>>) -> Self {
        assert!(
            layers.last().expect("layer stack empty").layer_type() == "OutputProjection",
            "letzte Schicht muss OutputProjection sein"
        );
        Self {
            tokenizer,
            network: layers,
        }
    }

    // ------------------------------------------------------------------------
    /// Liefert eine menschenlesbare Beschreibung des Layer-Stacks.
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|l| l.layer_type())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // ------------------------------------------------------------------------
    /// Aggregiert die Gesamtzahl der lernbaren Parameter.
    pub fn total_parameters(&self) -> usize {
        self.network.iter().map(|l| l.parameters()).sum()
    }

    // ------------------------------------------------------------------------
    // PUBLIC API  –  Inferenz
    // ------------------------------------------------------------------------
    /**
     *  Führt eine vollständige Vorwärtspropagation einschließlich
     *  autoregressiver Token‐Generierung durch und dekodiert die resultierende
     *  Token-Sequenz zu UTF-8.
     */
    pub fn predict(&mut self, s_input: &str) -> String {
        let v_token_out = self.forward(s_input);
        if v_token_out.is_empty() {
            return String::new();
        }
        self.tokenizer.decode_tokens(&v_token_out)
    }

    // =======================================================================
    //  Funktion : forward()
    //  Modul    : llm.rs
    //  Autor    : Marcus Schlieper (ExpChat.ai)
    // -----------------------------------------------------------------------
    //  Beschreibung
    //  ------------
    //  Führt einen vollständigen Vorwärtsdurchlauf durch.  Das Verfahren
    //  kombiniert  (a)  Sliding-Window-Verarbeitung für Sequenzen, die länger
    //  als MAX_SEQ_LEN sind,  sowie  (b)  eine autoregressive Generierung
    //  zusätzlicher Tokens bis zum Erreichen des EOS-Symbols oder der
    //  Fenster­größe.  Der Ergebnis-Puffer wird initial leer gehalten, um ein
    //  reines Echo der Eingabe zu vermeiden.
    //
    //  Rückgabe
    //  --------
    //  Vec<usize> – generierte Token-IDs ohne Prompt-Echo.
    //
    // =======================================================================
    fn forward(&mut self, s_input: &str) -> Vec<usize> {
        let v_prompt_tokens: Vec<usize> = self.tokenizer.encode_text(s_input);
        let v_chunks: Vec<Vec<usize>> = chunk_sequence(&v_prompt_tokens, MAX_SEQ_LEN / 5);

        let mut v_output_global: Vec<usize> = Vec::new();
        let i_eos: usize = self.tokenizer.eos_id(); // <-- stelle sicher, dass es diese API gibt

        for (i_chunk_idx, v_chunk) in v_chunks.iter().enumerate() {
            let mut v_context_tokens: Vec<usize> = v_chunk.clone();
            let mut v_generated_tokens: Vec<usize> = Vec::new();

            for _ in 0..(MAX_SEQ_LEN - v_context_tokens.len()) {
                let a_input = ndarray::Array2::from_shape_vec(
                    (1, v_context_tokens.len()),
                    v_context_tokens.iter().map(|&id| id as f32).collect(),
                )
                .expect("ungueltige Tensorform fuer Eingabe");

                let mut a_activ = a_input;
                for layer in &mut self.network {
                    a_activ = layer.forward(&a_activ);
                }

                let i_last_row = a_activ.shape()[0] - 1;
                let a_last = a_activ
                    .row(i_last_row)
                    .to_owned()
                    .insert_axis(ndarray::Axis(0));
                let a_probs = Self::softmax(&a_last);

                let i_next: usize = Self::greedy_decode(&a_probs)
                    .last()
                    .copied()
                    .expect("leere Greedy-Decoding-Ausgabe");

                if i_next == i_eos {
                    break;
                }

                v_context_tokens.push(i_next);
                v_generated_tokens.push(i_next);

                if v_context_tokens.len() >= MAX_SEQ_LEN {
                    break;
                }
            }

            if i_chunk_idx + 1 < v_chunks.len() {
                if let Some(&last) = v_generated_tokens.last() {
                    if last == i_eos {
                        v_generated_tokens.pop();
                    }
                }
            }
            v_output_global.extend_from_slice(&v_generated_tokens);
        }
        v_output_global
    }

    // ------------------------------------------------------------------------
    // train  –  Sliding-Window-Training mit Cross-Entropy
    // ------------------------------------------------------------------------
    /**
     *  Führt Mini-Batch-ähnliches Training über ein Sliding-Window-Schema aus.
     *  Die Gewichte aller Schichten werden mittels Adam optimiert.
     */
    pub fn train(&mut self, v_texts: Vec<&str>, i_epochs: usize, d_lr: f32) {
        // 1) Stop-Flag (Ctrl+C oder Taste 'q')
        let stop_flag = Arc::new(AtomicBool::new(false));
        {
            let stop_flag_ctrlc = std::sync::Arc::clone(&stop_flag);
            match ctrlc::set_handler(move || {
                stop_flag_ctrlc.store(true, AtomicOrdering::SeqCst);
            }) {
                Ok(()) => {}
                Err(ctrlc::Error::MultipleHandlers) => {
                    // Handler war schon gesetzt. Alles okay. Weiter machen.
                }
                Err(e) => {
                    eprintln!("Ctrl+C-Handler Warnung: {e}");
                }
            }
        }

        // 2) Dataset einmal tokenisieren (wie gehabt)
        let v_tokenized: Vec<Vec<usize>> = v_texts
            .iter()
            .map(|s| self.tokenizer.encode_text(s))
            .collect();

        for i_epoch in 0..i_epochs {
            let t_epoch_start = Instant::now();
            let mut d_total_loss: f32 = 0.0;
            let mut i_steps: usize = 0;
            let mut i_tokens_epoch: usize = 0;

            // Hinweis: jederzeit abbrechbar
            if stop_flag.load(AtomicOrdering::Relaxed) {
                println!("Training abgebrochen (Ctrl+C)");
                return;
            }

            for v_sample in &v_tokenized {
                // Tastendruck 'q' (nicht blockierend)
                if event::poll(Duration::from_millis(0)).unwrap_or(false) {
                    if let Ok(Event::Key(key)) = event::read() {
                        if key.kind == KeyEventKind::Press {
                            if matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q')) {
                                println!("Training abgebrochen (Taste 'q')");
                                return;
                            }
                        }
                    }
                }
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    println!("Training abgebrochen (Ctrl+C)");
                    return;
                }

                let v_chunks =
                    crate::utils::chunk_sequence(v_sample, crate::config::MAX_SEQ_LEN / 5);
                for v_chunk in v_chunks {
                    if v_chunk.len() < 2 {
                        continue; // zu kurz für Target-Shift
                    }
                    let v_input_ids = &v_chunk[..v_chunk.len() - 1];
                    let v_target_ids = &v_chunk[1..];

                    // 3) Eingabetensor
                    let a_ids: ndarray::Array1<f32> =
                        ndarray::Array1::from_iter(v_input_ids.iter().map(|&id| id as f32));
                    let mut a_input: ndarray::Array2<f32> =
                        ndarray::Array2::zeros((1, v_input_ids.len()));
                    a_input.row_mut(0).assign(&a_ids);

                    // 4) Forward-Pass
                    let mut a_forward = a_input;
                    for layer in &mut self.network {
                        a_forward = layer.forward(&a_forward);
                    }

                    // 5) Loss
                    let a_probs = Self::softmax(&a_forward);
                    d_total_loss += Self::cross_entropy_loss_step(&a_probs, v_target_ids);

                    // 6) Gradienten
                    let mut a_grads = Self::compute_gradients_step(&a_probs, v_target_ids);
                    Self::clip_gradients(&mut a_grads, 5.0);

                    // 7) Backward-Pass
                    for layer in self.network.iter_mut().rev() {
                        a_grads = layer.backward(&a_grads, d_lr);
                    }

                    // 8) Zähler
                    i_steps += 1;
                    i_tokens_epoch += v_target_ids.len();

                    // 9) Optional: häufiger Abbruch-Check
                    if event::poll(Duration::from_millis(0)).unwrap_or(false) {
                        if let Ok(Event::Key(key)) = event::read() {
                            if key.kind == KeyEventKind::Press {
                                if matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q')) {
                                    println!("Training abgebrochen (Taste 'q')");
                                    return;
                                }
                            }
                        }
                    }
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        println!("Training abgebrochen (Ctrl+C)");
                        return;
                    }
                }
            }

            // 10) Epochen-Statistik: Loss + Tokens/Sekunde
            let secs = t_epoch_start.elapsed().as_secs_f32().max(1e-6);
            let d_avg_loss = if i_steps > 0 {
                d_total_loss / (i_steps as f32)
            } else {
                0.0
            };
            let tps = (i_tokens_epoch as f32) / secs;

            println!(
                "Epoch {}  Loss {:.4}  Tokens/s {:.0}  (Tokens: {}, Dauer: {:.2}s)",
                i_epoch, d_avg_loss, tps, i_tokens_epoch, secs
            );

            // Direkt hier abbrechen, falls Taste gedrückt
            if stop_flag.load(AtomicOrdering::Relaxed) {
                println!("Training abgebrochen nach Epoch {}", i_epoch);
                return;
            }
            if event::poll(Duration::from_millis(0)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        if matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q')) {
                            println!("Training abgebrochen nach Epoch {} (Taste 'q')", i_epoch);
                            return;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // ------------------------- Mathematische Hilfen -------------------------
    // ------------------------------------------------------------------------
    /// Numerisch stabile Soft-max-Funktion (zeilenweise).
    fn softmax(a_logits: &Array2<f32>) -> Array2<f32> {
        let mut a_result = a_logits.clone();
        for mut row in a_result.rows_mut() {
            let d_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let v_exp: Vec<f32> = row.iter().map(|&x| (x - d_max).exp()).collect();
            let d_sum: f32 = v_exp.iter().sum();
            for (i_idx, &d_val) in v_exp.iter().enumerate() {
                row[i_idx] = d_val / d_sum;
            }
        }
        a_result
    }

    /// Greedy-Decoding: maximal wahrscheinlicher Index pro Zeile.
    fn greedy_decode(a_probs: &Array2<f32>) -> Vec<usize> {
        a_probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(CmpOrdering::Equal))
                    .map(|(idx, _)| idx)
                    .expect("row empty")
            })
            .to_vec()
    }

    /// Kreuzentropie-Verlust für ein Fenster
    fn cross_entropy_loss_step(a_probs: &Array2<f32>, v_target: &[usize]) -> f32 {
        let mut d_loss = 0.0;
        for (i_row, &i_tgt) in v_target.iter().enumerate() {
            let d_p = a_probs[(i_row, i_tgt)].max(1e-15);
            d_loss -= d_p.ln();
        }
        d_loss / v_target.len() as f32
    }

    /// Ableitung der Kreuzentropie (Soft-max integriert)
    fn compute_gradients_step(a_probs: &Array2<f32>, v_target: &[usize]) -> Array2<f32> {
        let mut a_grad = a_probs.clone();
        for (i_row, &i_tgt) in v_target.iter().enumerate() {
            a_grad[(i_row, i_tgt)] -= 1.0;
        }
        let d_batch = v_target.len() as f32;
        a_grad.mapv(|x| x / d_batch)
    }

    /// Globales Gradient-Clipping (L2-Norm)
    fn clip_gradients(a_grads: &mut ndarray::Array2<f32>, d_max_norm: f32) {
        let d_norm: f32 = a_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if d_norm > d_max_norm {
            let d_scale = d_max_norm / d_norm;
            a_grads.mapv_inplace(|x| x * d_scale);
        }
    }

    // Speichern (mit Cache-Clear und BufWriter)
    pub fn save_checkpoint(&mut self, s_path: &str) -> Result<()> {
        // 1) Attention-Caches leeren (kleinerer Checkpoint)
        for layer in &mut self.network {
            if let Some(block) = layer.as_any_mut().downcast_mut::<TransformerBlockV2>() {
                block.attention.clear_cache();
            }
        }

        // 2) Gepuffert schreiben
        let f = File::create(s_path)
            .with_context(|| format!("Kann Datei {} nicht erstellen", s_path))?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f); // 8 MB
        let cfg = config::standard();

        // Hinweis: Tokenizer nicht mitspeichern (liegt separat in data/…)
        // encode_into_std_write(&self.tokenizer, &mut w, cfg)?; // bewusst weggelassen

        // 3) Nur Layer-Gewichte serialisieren
        for layer in &self.network {
            let any = layer.as_any();
            if any.is::<Embeddings>() {
                encode_into_std_write(any.downcast_ref::<Embeddings>().unwrap(), &mut w, cfg)?;
            } else if any.is::<TransformerBlockV2>() {
                encode_into_std_write(
                    any.downcast_ref::<TransformerBlockV2>().unwrap(),
                    &mut w,
                    cfg,
                )?;
            } else if any.is::<OutputProjection>() {
                encode_into_std_write(
                    any.downcast_ref::<OutputProjection>().unwrap(),
                    &mut w,
                    cfg,
                )?;
            } else {
                unreachable!("unbekannter Layer-Typ");
            }
        }

        w.flush().ok(); // sauber flushen
        Ok(())
    }

    /// Deserialisiert Tokenizer + Layer-Stack; bei fehlender Datei bleibt das
    /// Modell untrainiert, andernfalls werden die gespeicherten Gewichte
    /// vollständig wiederhergestellt.
    pub fn load_checkpoint(&mut self, s_path: &str) -> Result<()> {
        use std::io::ErrorKind;

        let f = match File::open(s_path) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("Checkpoint {} nicht gefunden, untrainiertes Modell", s_path);
                return Ok(());
            }
            Err(e) => return Err(e).context("Fehler beim Öffnen des Checkpoints"),
        };

        let mut r = BufReader::with_capacity(8 * 1024 * 1024, f); // 8 MB
        let cfg = config::standard();

        // Hinweis: Tokenizer wird hier nicht gelesen (liegt separat in data/…)
        // self.tokenizer = decode_from_std_read(&mut r, cfg)
        //     .context("Fehler beim Deserialisieren des Tokenizers")?;

        for layer in &mut self.network {
            let any = layer.as_any_mut();
            if any.is::<Embeddings>() {
                *any.downcast_mut::<Embeddings>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else if any.is::<TransformerBlockV2>() {
                *any.downcast_mut::<TransformerBlockV2>().unwrap() =
                    decode_from_std_read(&mut r, cfg)?;
            } else if any.is::<OutputProjection>() {
                *any.downcast_mut::<OutputProjection>().unwrap() =
                    decode_from_std_read(&mut r, cfg)?;
            } else {
                unreachable!("unbekannter Layer-Typ beim Laden");
            }
        }

        // Nach dem Laden: Attention-Caches leeren, damit sie frisch sind
        for layer in &mut self.network {
            if let Some(block) = layer.as_any_mut().downcast_mut::<TransformerBlockV2>() {
                block.attention.clear_cache();
            }
        }
        println!("Checkpoint loaded");
        Ok(())
    }
}

// ---------------------- Default-Implementierung -----------------------------
/**
 *  Stellt eine gebrauchsfertige Minimal-Konfiguration bereit:
 *
 *      Embeddings ▸ Transformer ▸ Transformer ▸ OutputProjection
 *
 *  Der Tokenizer ist ein reiner Byte-Level-Tokenizer und erfordert somit
 *  keinerlei externes Training, um prinzipiell lauffähig zu sein.
 */
impl Default for LLM {
    fn default() -> Self {
        let tokenizer = Tokenizer::new_byte_level();
        let i_vocab_size = tokenizer.vocab_size();

        let embeddings = Box::new(Embeddings::from_tokenizer(&tokenizer));
        let transformer1 = Box::new(TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, 8, 0.1));
        let transformer2 = Box::new(TransformerBlockV2::new(EMBEDDING_DIM, HIDDEN_DIM, 8, 0.1));
        let output_layer = Box::new(OutputProjection::new(EMBEDDING_DIM, i_vocab_size));

        Self::new(
            tokenizer,
            vec![embeddings, transformer1, transformer2, output_layer],
        )
    }
}
