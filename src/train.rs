// train.rs
// ============================================================================
// Autor:   Marcus Schlieper (ExpChat.ai)
// Hinweis: LLM Orchestrierung: Netzwerk, Training, Inferenz, Checkpoints.
// ============================================================================

#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use bincode::{config, decode_from_std_read, encode_into_std_write};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ndarray::{Array1, Array2, Axis};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;
use std::time::Instant;

use std::io::{self, Write};

use crate::layers::{Layer, Embeddings, OutputProjection, TransformerBlockV2};
use crate::math::{softmax, cross_entropy_loss_step, compute_gradients_step, clip_gradients};
use crate::utils::{Tokenizer, MAX_SEQ_LEN, EMBEDDING_DIM, S_EOS, chunk_sequence};

pub struct LLM {
    pub tokenizer: Tokenizer,
    pub network: Vec<Box<dyn Layer>>,
}

impl LLM {
    pub fn new(tokenizer: Tokenizer, layers: Vec<Box<dyn Layer>>) -> Self {
        assert!(
            layers.last().expect("layer stack empty").layer_type() == "OutputProjection",
            "letzte Schicht muss OutputProjection sein"
        );
        Self { tokenizer, network: layers }
    }

    pub fn set_batch_accumulation(&mut self, batch_size: usize) {
    let bsz = batch_size.max(1);
    for layer in &mut self.network {
        let any = layer.as_any_mut();
        if let Some(op) = any.downcast_mut::<OutputProjection>() {
            op.set_accumulate_steps(bsz);
        } else if let Some(ln) = any.downcast_mut::<crate::layers::LayerNorm>() {
            ln.set_accumulate_steps(bsz);
        } else if let Some(block) = any.downcast_mut::<TransformerBlockV2>() {
            block.norm1.set_accumulate_steps(bsz);
            block.norm2.set_accumulate_steps(bsz);
            block.attention.set_accumulate_steps(bsz);
            block.feedforward.set_accumulate_steps(bsz);
        } else if let Some(embed) = any.downcast_mut::<crate::layers::Embeddings>() {
            embed.token_optimizer.set_accumulate_steps(bsz);
            embed.positional_optimizer.set_accumulate_steps(bsz);
        }
    }
}


    pub fn network_description(&self) -> String {
        self.network.iter().map(|l| l.layer_type()).collect::<Vec<_>>().join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        self.network.iter().map(|l| l.parameter_count()).sum()
    }

    pub fn predict(&mut self, s_input: &str) -> String {
        let v_token_out = self.forward(s_input);
        if v_token_out.is_empty() {
            return String::new();
        }
        self.tokenizer.decode_tokens(&v_token_out)
    }

    fn forward(&mut self, s_input: &str) -> Vec<usize> {
        let v_prompt_tokens: Vec<usize> = self.tokenizer.encode_text(s_input);
        let v_chunks: Vec<Vec<usize>> = chunk_sequence(&v_prompt_tokens, MAX_SEQ_LEN / 5);

        let mut v_output_global: Vec<usize> = Vec::new();
        let i_eos: usize = self.tokenizer.eos_id();

        for (i_chunk_idx, v_chunk) in v_chunks.iter().enumerate() {
            let mut v_context_tokens: Vec<usize> = v_chunk.clone();
            let mut v_generated_tokens: Vec<usize> = Vec::new();

            for _ in 0..(MAX_SEQ_LEN - v_context_tokens.len()) {
                let a_input = Array2::from_shape_vec(
                    (1, v_context_tokens.len()),
                    v_context_tokens.iter().map(|&id| id as f32).collect(),
                ).expect("invalid input shape");

                let mut a_activ = a_input;
                for layer in &mut self.network {
                    a_activ = layer.forward(&a_activ);
                }

                let i_last_row = a_activ.shape()[0] - 1;
                let a_last = a_activ.row(i_last_row).to_owned().insert_axis(Axis(0));
                let a_probs = softmax(&a_last);

                // greedy
                let (mut best_idx, mut best_val) = (0usize, f32::NEG_INFINITY);
                for (i, &p) in a_probs.row(0).iter().enumerate() {
                    if p > best_val {
                        best_val = p;
                        best_idx = i;
                    }
                }
                let i_next = best_idx;

                if i_next == i_eos {
                    break;
                }

                v_context_tokens.push(i_next);
                v_generated_tokens.push(i_next);

                // Streaming: Token sofort decodieren und anzeigen
                let s_piece: String = self.tokenizer.decode_tokens(&[i_next]);

                if s_piece.contains(S_EOS){
                    break;
                }

                if let Err(_e) = write!(io::stdout(), "{}]", s_piece) {
                    eprintln!("stdout write failed");
                }
                if let Err(_e) = io::stdout().flush() {
                    eprintln!("stdout flush failed");
                }

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
        
        let _ = writeln!(io::stdout(), "");

        v_output_global
    }

    pub fn train(&mut self, v_texts: Vec<&str>, i_epochs: usize, d_lr: f32, batch_size: usize) {
        self.set_batch_accumulation(batch_size);

        let stop_flag = Arc::new(AtomicBool::new(false));
        {
            let stop_flag_ctrlc = Arc::clone(&stop_flag);
            let _ = ctrlc::set_handler(move || {
                stop_flag_ctrlc.store(true, AtomicOrdering::SeqCst);
            });
        }

        let v_tokenized: Vec<Vec<usize>> = v_texts
            .iter()
            .map(|s| self.tokenizer.encode_text(s))
            .collect();

        for i_epoch in 0..i_epochs {
            let t_epoch_start = Instant::now();
            let mut d_total_loss: f32 = 0.0;
            let mut i_steps: usize = 0;
            let mut i_tokens_epoch: usize = 0;

            if stop_flag.load(AtomicOrdering::Relaxed) {
                println!("Training abgebrochen (Ctrl+C)");
                return;
            }

            for v_sample in &v_tokenized {
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

                let v_chunks = chunk_sequence(v_sample, MAX_SEQ_LEN / 5);
                for v_chunk in v_chunks {
                    if v_chunk.len() < 2 { continue; }
                    let v_input_ids = &v_chunk[..v_chunk.len() - 1];
                    let v_target_ids = &v_chunk[1..];

                    let a_ids: Array1<f32> = Array1::from_iter(v_input_ids.iter().map(|&id| id as f32));
                    let mut a_input: Array2<f32> = Array2::zeros((1, v_input_ids.len()));
                    a_input.row_mut(0).assign(&a_ids);

                    // println!("seq_in={}, tgt={}, vocab={}, embed={}", v_input_ids.len(), v_target_ids.len(), self.tokenizer.vocab_size(), EMBEDDING_DIM);

                    let mut a_forward = a_input;
                    for layer in &mut self.network {
                        a_forward = layer.forward(&a_forward);
                    }
                    assert_eq!(a_forward.shape()[0], v_target_ids.len(), "row mismatch");
                    assert_eq!(a_forward.shape()[1], self.tokenizer.vocab_size(), "cols != vocab size");

                    let a_probs = softmax(&a_forward);
                    d_total_loss += cross_entropy_loss_step(&a_probs, v_target_ids);

                    let mut a_grads = compute_gradients_step(&a_probs, v_target_ids);
                    clip_gradients(&mut a_grads, 1.0);

                    for layer in self.network.iter_mut().rev() {
                        a_grads = layer.backward(&a_grads, d_lr);
                    }

                    i_steps += 1;
                    i_tokens_epoch += v_target_ids.len();

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

            let secs = t_epoch_start.elapsed().as_secs_f32().max(1e-6);
            let d_avg_loss = if i_steps > 0 { d_total_loss / (i_steps as f32) } else { 0.0 };
            let tps = (i_tokens_epoch as f32) / secs;

            println!(
                "Epoch {}  Loss {:.4}  Tokens/s {:.0}  (Tokens: {}, Dauer: {:.2}s)",
                i_epoch, d_avg_loss, tps, i_tokens_epoch, secs
            );

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

    pub fn save_checkpoint(&mut self, s_path: &str) -> Result<()> {
        let f = File::create(s_path)
            .with_context(|| format!("Kann Datei {} nicht erstellen", s_path))?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f);
        let cfg = config::standard();

        for layer in &self.network {
            let any = layer.as_any();
            if any.is::<Embeddings>() {
                encode_into_std_write(any.downcast_ref::<Embeddings>().unwrap(), &mut w, cfg)?;
            } else if any.is::<TransformerBlockV2>() {
                encode_into_std_write(any.downcast_ref::<TransformerBlockV2>().unwrap(), &mut w, cfg)?;
            } else if any.is::<OutputProjection>() {
                encode_into_std_write(any.downcast_ref::<OutputProjection>().unwrap(), &mut w, cfg)?;
            } else {
                unreachable!("unbekannter Layer-Typ");
            }
        }
        w.flush().ok();
        Ok(())
    }

    pub fn load_checkpoint(&mut self, s_path: &str) -> Result<()> {
        use std::io::ErrorKind;

        let f = match File::open(s_path) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("Checkpoint {} nicht gefunden, untrainiertes Modell", s_path);
                return Ok(());
            }
            Err(e) => return Err(e).context("Fehler beim Oeffnen des Checkpoints"),
        };

        let mut r = BufReader::with_capacity(8 * 1024 * 1024, f);
        let cfg = config::standard();

        for layer in &mut self.network {
            let any = layer.as_any_mut();
            if any.is::<Embeddings>() {
                *any.downcast_mut::<Embeddings>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else if any.is::<TransformerBlockV2>() {
                *any.downcast_mut::<TransformerBlockV2>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else if any.is::<OutputProjection>() {
                *any.downcast_mut::<OutputProjection>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else {
                unreachable!("unbekannter Layer-Typ beim Laden");
            }
        }
        println!("Checkpoint loaded");
        Ok(())
    }
}