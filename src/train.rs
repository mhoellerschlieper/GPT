// train.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Training and inference for a small transformer style LLM.
//           Features:
//           - Stable softmax based training loop
//           - Two phase training (pretrain + main)
//           - Train/eval mode gating for dropout and stochastic depth
//           - Mixed precision style gradient scaling (f32 grads, dynamic scale)
//           - Loss masking for chat format: optimize only tokens after "Assistant:"
//           - Warmup + cosine learning rate schedule
//           - AdamW parameter groups via weight decay configuration:
//             disable decay for RMSNorm weights and biases (optional)
//           - Token based stop sequences for inference
//           - Sliding window inference for long contexts (up to MAX_SEQ_LEN_CANONICAL)
// History:
//  - 2026-01-16: Initial training and inference loop.
//  - 2026-01-17: Adds stable softmax and GradScaler, adds train/eval gating.
//  - 2026-01-18: Adds assistant-only loss masking, LR warmup+cosine schedule,
//                optimizer param groups, token stop sequences, sliding window inference.
// ============================================================================

#![forbid(unsafe_code)]
#![allow(warnings)]

use anyhow::{anyhow, Context, Result};
use bincode::{config, decode_from_std_read, encode_into_std_write};
use ndarray::{Array2, Axis};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use crate::layers::{Embeddings, Layer, OutputProjection, TransformerBlockV2};
use crate::math::{clip_gradients, softmax_stable, GradScaler};
use crate::tokenize::Tokenizer;
use crate::utils::MAX_SEQ_LEN_CANONICAL;

// ---------------------------------------------------------------------------
// Sampling params and inference grid
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub d_temperature: f32,
    pub i_k_top: usize,
    pub d_p_top: f32,
    pub b_greedy: bool,

    // Stop sequences that cause generation to stop if produced as a suffix.
    // These are matched on token ids, not on raw strings, to avoid partial matches.
    pub v_stop_sequences: Vec<String>,
}

impl SamplingParams {
    pub fn sane_default() -> Self {
        Self {
            d_temperature: 0.9,
            i_k_top: 50,
            d_p_top: 0.95,
            b_greedy: false,
            // Conservative defaults for chat like prompts.
            v_stop_sequences: vec!["\nUser:".to_string(), "\nAssistant:".to_string()],
        }
    }

    pub fn greedy_debug() -> Self {
        Self {
            d_temperature: 1.0,
            i_k_top: 0,
            d_p_top: 0.0,
            b_greedy: true,
            v_stop_sequences: vec!["\nUser:".to_string(), "\nAssistant:".to_string()],
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferenceGrid {
    pub v_temperatures: Vec<f32>,
    pub v_k_top: Vec<usize>,
    pub v_p_top: Vec<f32>,
}

impl InferenceGrid {
    pub fn conservative_default() -> Self {
        Self {
            v_temperatures: vec![0.7, 0.9, 1.0],
            v_k_top: vec![0, 20, 50],
            v_p_top: vec![0.85, 0.95, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub i_max_seq_len: usize,
    pub i_min_window_len: usize,

    pub d_lr: f32,
    pub i_batch_size: usize,
    pub d_grad_clip: f32,

    pub d_val_ratio: f32,

    pub i_epochs: usize,
    pub i_steps_per_epoch: usize,

    // Regularization controls
    pub d_attention_dropout: f32,
    pub d_residual_dropout: f32,
    pub d_stochastic_depth_p: f32,

    // Mixed precision style controls
    pub b_grad_scaling: bool,

    // Loss masking controls
    pub b_loss_mask_assistant_only: bool,
    pub s_assistant_marker: String,

    // LR schedule controls
    pub b_lr_schedule: bool,
    pub i_warmup_steps: usize,

    // AdamW parameter group approximation by weight decay configuration
    pub d_weight_decay_default: f32,
    pub b_no_weight_decay_for_norm_bias: bool,
}

impl TrainConfig {
    pub fn default_for_debug() -> Self {
        Self {
            i_max_seq_len: MAX_SEQ_LEN_CANONICAL,
            i_min_window_len: 32,
            d_lr: 1e-4,
            i_batch_size: 16,
            d_grad_clip: 1.0,
            d_val_ratio: 0.10,
            i_epochs: 10,
            i_steps_per_epoch: 200,
            d_attention_dropout: 0.1,
            d_residual_dropout: 0.0,
            d_stochastic_depth_p: 0.0,
            b_grad_scaling: true,

            b_loss_mask_assistant_only: true,
            s_assistant_marker: "Assistant:".to_string(),

            b_lr_schedule: true,
            i_warmup_steps: 200,

            d_weight_decay_default: 0.01,
            b_no_weight_decay_for_norm_bias: true,
        }
    }

    pub fn sanitize(&mut self) {
        if self.i_max_seq_len > MAX_SEQ_LEN_CANONICAL {
            self.i_max_seq_len = MAX_SEQ_LEN_CANONICAL;
        }
        if self.i_max_seq_len < 8 {
            self.i_max_seq_len = 8;
        }

        self.i_min_window_len = self.i_min_window_len.clamp(2, self.i_max_seq_len);
        self.i_batch_size = self.i_batch_size.max(1);

        self.d_val_ratio = self.d_val_ratio.clamp(0.0, 0.5);
        self.i_epochs = self.i_epochs.max(1);
        self.i_steps_per_epoch = self.i_steps_per_epoch.max(1);

        if !self.d_lr.is_finite() || self.d_lr <= 0.0 {
            self.d_lr = 1e-4;
        }
        if !self.d_grad_clip.is_finite() || self.d_grad_clip <= 0.0 {
            self.d_grad_clip = 1.0;
        }

        self.d_attention_dropout = self.d_attention_dropout.clamp(0.0, 0.9);
        self.d_residual_dropout = self.d_residual_dropout.clamp(0.0, 0.9);
        self.d_stochastic_depth_p = self.d_stochastic_depth_p.clamp(0.0, 0.9);

        self.i_warmup_steps = self.i_warmup_steps.max(1);
        self.d_weight_decay_default = self.d_weight_decay_default.max(0.0);

        if self.s_assistant_marker.trim().is_empty() {
            self.s_assistant_marker = "Assistant:".to_string();
        }
    }
}

// ---------------------------------------------------------------------------
// LLM container
// ---------------------------------------------------------------------------

pub struct LLM {
    pub tokenizer: Tokenizer,
    pub network: Vec<Box<dyn Layer>>,
    pub i_max_seq_len_runtime: usize,
}

impl LLM {
    pub fn new(tokenizer: Tokenizer, layers: Vec<Box<dyn Layer>>, i_max_seq_len_runtime: usize) -> Self {
        assert!(!layers.is_empty(), "layer stack empty");
        assert!(
            layers.last().unwrap().layer_type() == "OutputProjection",
            "last layer must be OutputProjection"
        );

        let i_max = i_max_seq_len_runtime.min(MAX_SEQ_LEN_CANONICAL).max(8);

        Self {
            tokenizer,
            network: layers,
            i_max_seq_len_runtime: i_max,
        }
    }

    // History:
    //  - 2026-01-17: Restores textual network description used by CLI.
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|l| l.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    // History:
    //  - 2026-01-17: Restores parameter counting used by CLI model info.
    pub fn total_parameters(&self) -> usize {
        self.network.iter().map(|l| l.parameter_count()).sum()
    }

    // History:
    //  - 2026-01-17: Restores inference grid runner used by CLI.
    pub fn run_inference_grid(&mut self, v_prompts: &[String], grid: &InferenceGrid) {
        self.set_eval_mode_no_dropout();

        for s_prompt in v_prompts {
            println!("PROMPT: {}", s_prompt);

            println!("GREEDY:");
            let _ = self.predict_greedy(s_prompt);

            for &d_temp in &grid.v_temperatures {
                for &i_k in &grid.v_k_top {
                    for &d_p in &grid.v_p_top {
                        let params = SamplingParams {
                            d_temperature: d_temp,
                            i_k_top: i_k,
                            d_p_top: d_p,
                            b_greedy: false,
                            v_stop_sequences: vec!["\nUser:".to_string(), "\nAssistant:".to_string()],
                        };

                        println!("SAMPLE: temp={:.2} k_top={} p_top={:.2}", d_temp, i_k, d_p);

                        let _ = self.generate_sliding_window(s_prompt, &params, true);
                    }
                }
            }

            println!("");
        }
    }

    pub fn set_eval_mode_no_dropout(&mut self) {
        self.set_train_mode(false);
    }

    pub fn set_train_mode(&mut self, b_train: bool) {
        for layer in &mut self.network {
            let any = layer.as_any_mut();
            if let Some(block) = any.downcast_mut::<TransformerBlockV2>() {
                block.set_train_mode(b_train);
            }
        }
    }

    pub fn set_batch_accumulation(&mut self, i_batch_size: usize) {
        let i_steps = i_batch_size.max(1);
        for layer in &mut self.network {
            let any = layer.as_any_mut();
            if let Some(op) = any.downcast_mut::<OutputProjection>() {
                op.set_accumulate_steps(i_steps);
            } else if let Some(block) = any.downcast_mut::<TransformerBlockV2>() {
                block.set_accumulate_steps(i_steps);
            } else if let Some(embed) = any.downcast_mut::<Embeddings>() {
                embed.set_accumulate_steps(i_steps);
            }
        }
    }

    pub fn apply_regularization_cfg(&mut self, cfg: &TrainConfig) {
        for layer in &mut self.network {
            let any = layer.as_any_mut();
            if let Some(block) = any.downcast_mut::<TransformerBlockV2>() {
                block.set_regularization(cfg.d_attention_dropout, cfg.d_residual_dropout, cfg.d_stochastic_depth_p);
            }
        }
    }

    pub fn apply_optimizer_param_groups(&mut self, cfg: &TrainConfig) {
        // NOTE:
        // This requires optimizer fields in layers.rs to be public:
        // - MultiHeadAttention: opt_qkv, opt_o
        // - FeedForwardGeGLU: opt_w_in, opt_w_out
        // - RMSNorm: opt_w
        //
        // Policy:
        // - Apply default weight decay to weight matrices
        // - Disable weight decay for biases and norm weights if configured

        for layer in &mut self.network {
            let any = layer.as_any_mut();

            if let Some(embed) = any.downcast_mut::<Embeddings>() {
                embed.token_optimizer.set_weight_decay(cfg.d_weight_decay_default);
                embed.positional_optimizer.set_weight_decay(cfg.d_weight_decay_default);
                continue;
            }

            if let Some(op) = any.downcast_mut::<OutputProjection>() {
                op.opt_w.set_weight_decay(cfg.d_weight_decay_default);
                if cfg.b_no_weight_decay_for_norm_bias {
                    op.opt_b.set_weight_decay(0.0);
                } else {
                    op.opt_b.set_weight_decay(cfg.d_weight_decay_default);
                }
                continue;
            }

            if let Some(block) = any.downcast_mut::<TransformerBlockV2>() {
                block.attention.opt_qkv.set_weight_decay(cfg.d_weight_decay_default);
                block.attention.opt_o.set_weight_decay(cfg.d_weight_decay_default);

                block.feedforward.opt_w_in.set_weight_decay(cfg.d_weight_decay_default);
                block.feedforward.opt_w_out.set_weight_decay(cfg.d_weight_decay_default);

                if cfg.b_no_weight_decay_for_norm_bias {
                    block.norm1.opt_w.set_weight_decay(0.0);
                    block.norm2.opt_w.set_weight_decay(0.0);
                } else {
                    block.norm1.opt_w.set_weight_decay(cfg.d_weight_decay_default);
                    block.norm2.opt_w.set_weight_decay(cfg.d_weight_decay_default);
                }

                continue;
            }
        }
    }

    pub fn predict(&mut self, s_input: &str) -> String {
        self.set_eval_mode_no_dropout();
        let params = SamplingParams::sane_default();
        let v_ids = self.generate_sliding_window(s_input, &params, true);
        self.tokenizer.decode_tokens(&v_ids)
    }

    pub fn predict_greedy(&mut self, s_input: &str) -> String {
        self.set_eval_mode_no_dropout();
        let params = SamplingParams::greedy_debug();
        let v_ids = self.generate_sliding_window(s_input, &params, true);
        self.tokenizer.decode_tokens(&v_ids)
    }

    // Robust generation with:
    // - sliding window truncation on context overflow
    // - token based stop sequences
    fn generate_sliding_window(&mut self, s_input: &str, params: &SamplingParams, b_stream: bool) -> Vec<usize> {
        let mut v_context: Vec<usize> = self.tokenizer.encode_text(s_input);
        let i_eos = self.tokenizer.eos_id();

        let i_cap = self.i_max_seq_len_runtime.min(MAX_SEQ_LEN_CANONICAL).max(8);

        // Precompute stop sequences as token ids without EOS.
        let v_stop_tok: Vec<Vec<usize>> = params
            .v_stop_sequences
            .iter()
            .map(|s| self.tokenizer.encode_stop_sequence(s))
            .filter(|v| !v.is_empty())
            .collect();

        if v_context.len() >= i_cap {
            let i_keep = i_cap.saturating_sub(1).max(1);
            v_context = v_context[v_context.len().saturating_sub(i_keep)..].to_vec();
        }

        let mut v_out: Vec<usize> = Vec::new();
        let i_budget = i_cap.saturating_sub(v_context.len());

        for _ in 0..i_budget {
            if v_context.is_empty() {
                break;
            }

            let a_input = Array2::from_shape_vec(
                (1, v_context.len()),
                v_context.iter().map(|&id| id as f32).collect(),
            )
            .expect("invalid input shape");

            let mut a_activ = a_input;
            for layer in &mut self.network {
                a_activ = layer.forward(&a_activ);
            }

            if a_activ.nrows() == 0 {
                break;
            }

            let i_last_row = a_activ.nrows() - 1;
            let a_last = a_activ.row(i_last_row).to_owned().insert_axis(Axis(0));

            let mut a_probs = softmax_stable(&a_last);

            let i_next = if params.b_greedy {
                argmax_row0(&a_probs)
            } else {
                apply_temperature_inplace(&mut a_probs, params.d_temperature);
                sample_k_top_p_top(&a_probs, params.i_k_top, params.d_p_top)
            };

            if i_next == i_eos {
                break;
            }

            v_context.push(i_next);
            v_out.push(i_next);

            if ends_with_any(&v_out, &v_stop_tok) {
                break;
            }

            if v_context.len() >= i_cap {
                // Sliding window: keep last (cap - 1) tokens for next forward pass.
                let i_keep = i_cap.saturating_sub(1).max(1);
                v_context = v_context[v_context.len().saturating_sub(i_keep)..].to_vec();
            }

            if b_stream {
                let s_piece = self.tokenizer.decode_tokens(&[i_next]);
                let _ = write!(io::stdout(), "{}", s_piece);
                let _ = io::stdout().flush();
            }
        }

        if b_stream {
            let _ = writeln!(io::stdout(), "");
        }

        v_out
    }

    pub fn train_two_phase(
        &mut self,
        v_pretrain_texts: &[String],
        v_main_texts: &[String],
        cfg_pretrain: &TrainConfig,
        cfg_main: &TrainConfig,
    ) {
        println!("PHASE A: pretraining");
        self.train_with_validation(v_pretrain_texts, cfg_pretrain);

        println!("PHASE B: main training");
        self.train_with_validation(v_main_texts, cfg_main);
    }

    pub fn train_with_validation(&mut self, v_texts: &[String], cfg_in: &TrainConfig) {
        let mut cfg = cfg_in.clone();
        cfg.sanitize();

        self.i_max_seq_len_runtime = cfg.i_max_seq_len.min(MAX_SEQ_LEN_CANONICAL).max(8);
        self.set_batch_accumulation(cfg.i_batch_size);
        self.apply_regularization_cfg(&cfg);
        self.apply_optimizer_param_groups(&cfg);
        self.set_train_mode(true);

        let mut grad_scaler = GradScaler::new_default();

        let stop_flag = Arc::new(AtomicBool::new(false));
        {
            let stop_flag_ctrlc = Arc::clone(&stop_flag);
            let _ = ctrlc::set_handler(move || {
                stop_flag_ctrlc.store(true, AtomicOrdering::SeqCst);
            });
        }

        // NOTE:
        // Grouped split helps reduce leakage if corpus pipeline prefixes samples with "[src=...]".
        let (v_train_texts, v_val_texts) = split_train_val_by_prefix_group(v_texts, cfg.d_val_ratio);

        println!(
            "TRAIN: samples={} VAL: samples={} max_seq_len={} base_lr={:.8} steps_per_epoch={} batch_size={} masked_loss={} warmup_steps={} wd={:.4}",
            v_train_texts.len(),
            v_val_texts.len(),
            cfg.i_max_seq_len,
            cfg.d_lr,
            cfg.i_steps_per_epoch,
            cfg.i_batch_size,
            cfg.b_loss_mask_assistant_only,
            cfg.i_warmup_steps,
            cfg.d_weight_decay_default
        );

        let mut i_global_step: usize = 0;
        let i_total_steps: usize = cfg.i_epochs.saturating_mul(cfg.i_steps_per_epoch).max(1);

        for i_epoch in 0..cfg.i_epochs {
            if stop_flag.load(AtomicOrdering::Relaxed) {
                println!("Training aborted (Ctrl+C)");
                return;
            }

            let t_start = Instant::now();
            let mut d_loss_sum: f32 = 0.0;
            let mut i_steps: usize = 0;
            let mut i_tokens: usize = 0;

            for _ in 0..cfg.i_steps_per_epoch {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    println!("Training aborted (Ctrl+C)");
                    return;
                }

                let (v_in, v_tgt, v_mask) = sample_random_window_pair_from_texts_masked(
                    &self.tokenizer,
                    &v_train_texts,
                    cfg.i_max_seq_len,
                    cfg.i_min_window_len,
                    cfg.b_loss_mask_assistant_only,
                    &cfg.s_assistant_marker,
                );

                if v_in.is_empty() || v_tgt.is_empty() {
                    continue;
                }

                let a_input = Array2::from_shape_vec((1, v_in.len()), v_in.iter().map(|&id| id as f32).collect())
                    .expect("invalid input shape");

                let mut a_forward = a_input;
                for layer in &mut self.network {
                    a_forward = layer.forward(&a_forward);
                }

                if a_forward.nrows() != v_tgt.len() {
                    continue;
                }

                let a_probs = softmax_stable(&a_forward);

                let d_step_loss = masked_cross_entropy_loss_step(&a_probs, &v_tgt, &v_mask);
                d_loss_sum += d_step_loss;

                let mut a_grads = masked_compute_gradients_step(&a_probs, &v_tgt, &v_mask);

                if cfg.b_grad_scaling {
                    grad_scaler.scale_grads_inplace(&mut a_grads);
                }

                if cfg.b_grad_scaling && !GradScaler::grads_are_finite(&a_grads) {
                    grad_scaler.update(true);
                    continue;
                }

                if cfg.b_grad_scaling {
                    grad_scaler.unscale_grads_inplace(&mut a_grads);
                }

                clip_gradients(&mut a_grads, cfg.d_grad_clip);

                // LR schedule per global step
                let d_lr_eff = if cfg.b_lr_schedule {
                    lr_warmup_cosine(cfg.d_lr, i_global_step, i_total_steps, cfg.i_warmup_steps)
                } else {
                    cfg.d_lr
                };

                for layer in self.network.iter_mut().rev() {
                    a_grads = layer.backward(&a_grads, d_lr_eff);
                }

                i_steps += 1;
                i_tokens += v_tgt.len();
                i_global_step = i_global_step.saturating_add(1);

                if cfg.b_grad_scaling {
                    grad_scaler.update(false);
                }
            }

            // Validation should run in eval mode without dropout
            self.set_eval_mode_no_dropout();
            let (d_val_loss, d_val_acc) = evaluate_validation_masked(self, &v_val_texts, &cfg);

            let d_train_loss = if i_steps > 0 { d_loss_sum / (i_steps as f32) } else { 0.0 };
            let d_secs = t_start.elapsed().as_secs_f32().max(1e-6);
            let d_tps = (i_tokens as f32) / d_secs;

            println!(
                "Epoch {} train_loss {:.4} val_loss {:.4} val_acc {:.4} tokens_s {:.0} steps {}",
                i_epoch, d_train_loss, d_val_loss, d_val_acc, d_tps, i_steps
            );

            // Restore train mode for next epoch
            self.set_train_mode(true);
        }
    }

    pub fn save_checkpoint(&mut self, s_path: &str) -> Result<()> {
        let f = File::create(s_path).with_context(|| format!("cannot create file: {}", s_path))?;
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
                unreachable!("unknown layer type during save");
            }
        }

        let _ = w.flush();
        Ok(())
    }

    pub fn load_checkpoint(&mut self, s_path: &str) -> Result<()> {
        use std::io::ErrorKind;

        let f = match File::open(s_path) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                eprintln!("checkpoint not found: {}, using untrained model", s_path);
                return Ok(());
            }
            Err(e) => return Err(e).context("failed to open checkpoint"),
        };

        let mut r = BufReader::with_capacity(8 * 1024 * 1024, f);
        let cfg = config::standard();

        for layer in &mut self.network {
            let any = layer.as_any_mut();

            if any.is::<Embeddings>() {
                let loaded: Embeddings = decode_from_std_read(&mut r, cfg)?;
                let i_rows = loaded.positional_embeddings.nrows();
                if i_rows != MAX_SEQ_LEN_CANONICAL {
                    return Err(anyhow!("checkpoint incompatible: positional_embeddings rows mismatch"));
                }
                *any.downcast_mut::<Embeddings>().unwrap() = loaded;
            } else if any.is::<TransformerBlockV2>() {
                *any.downcast_mut::<TransformerBlockV2>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else if any.is::<OutputProjection>() {
                *any.downcast_mut::<OutputProjection>().unwrap() = decode_from_std_read(&mut r, cfg)?;
            } else {
                unreachable!("unknown layer type during load");
            }
        }

        println!("checkpoint loaded");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Masked loss helpers
// ---------------------------------------------------------------------------

fn masked_cross_entropy_loss_step(a_probs: &Array2<f32>, v_target: &[usize], v_mask: &[u8]) -> f32 {
    // History:
    //  - 2026-01-18: Adds assistant-only masked cross entropy.
    if v_target.is_empty() {
        return 0.0;
    }

    let mut d_loss = 0.0f32;
    let mut d_count = 0.0f32;

    for (i_row, &i_tgt) in v_target.iter().enumerate() {
        if i_row >= a_probs.nrows() || i_tgt >= a_probs.ncols() {
            continue;
        }
        if i_row >= v_mask.len() || v_mask[i_row] == 0 {
            continue;
        }

        let d_p = a_probs[(i_row, i_tgt)].max(1e-15);
        d_loss -= d_p.ln();
        d_count += 1.0;
    }

    if d_count <= 0.0 {
        return 0.0;
    }

    d_loss / d_count
}

fn masked_compute_gradients_step(a_probs: &Array2<f32>, v_target: &[usize], v_mask: &[u8]) -> Array2<f32> {
    // Gradient equals (probs - one_hot(target)) / count on masked positions,
    // and 0 on masked-out positions.
    let mut a_grad = a_probs.clone();

    if v_target.is_empty() {
        return a_grad;
    }

    let mut d_count = 0.0f32;
    for i_row in 0..v_target.len().min(a_grad.nrows()) {
        if i_row < v_mask.len() && v_mask[i_row] == 1 {
            d_count += 1.0;
        }
    }

    if d_count <= 0.0 {
        a_grad.fill(0.0);
        return a_grad;
    }

    for (i_row, &i_tgt) in v_target.iter().enumerate() {
        if i_row >= a_grad.nrows() || i_tgt >= a_grad.ncols() {
            continue;
        }

        if i_row < v_mask.len() && v_mask[i_row] == 1 {
            a_grad[(i_row, i_tgt)] -= 1.0;
        } else {
            for j in 0..a_grad.ncols() {
                a_grad[(i_row, j)] = 0.0;
            }
        }
    }

    a_grad.mapv(|x| x / d_count.max(1.0))
}

// ---------------------------------------------------------------------------
// LR schedule
// ---------------------------------------------------------------------------

fn lr_warmup_cosine(d_base_lr: f32, i_step: usize, i_total_steps: usize, i_warmup_steps: usize) -> f32 {
    // History:
    //  - 2026-01-18: Adds warmup + cosine LR schedule.
    let d_lr = if d_base_lr.is_finite() { d_base_lr } else { 1e-4 };
    let i_total = i_total_steps.max(1);
    let i_warm = i_warmup_steps.max(1);

    if i_step < i_warm {
        let d = (i_step as f32) / (i_warm as f32);
        return d_lr * d.clamp(0.0, 1.0);
    }

    let i_rem = i_total.saturating_sub(i_warm).max(1);
    let i_pos = i_step.saturating_sub(i_warm).min(i_rem);
    let d_t = (i_pos as f32) / (i_rem as f32);

    let d_cos = (std::f32::consts::PI * d_t).cos();
    let d_decay = 0.5 * (1.0 + d_cos);

    let d_floor = 0.05f32; // 5 percent floor
    d_lr * (d_floor + (1.0 - d_floor) * d_decay)
}

// ---------------------------------------------------------------------------
// Validation (masked)
// ---------------------------------------------------------------------------

fn evaluate_validation_masked(llm: &mut LLM, v_val_texts: &[String], cfg: &TrainConfig) -> (f32, f32) {
    if v_val_texts.is_empty() {
        return (0.0, 0.0);
    }

    let mut d_loss_sum: f32 = 0.0;
    let mut i_steps: usize = 0;
    let mut i_top1_hits: usize = 0;
    let mut i_top1_total: usize = 0;

    let i_eval_steps = (cfg.i_steps_per_epoch / 4).max(50);

    for _ in 0..i_eval_steps {
        let (v_in, v_tgt, v_mask) = sample_random_window_pair_from_texts_masked(
            &llm.tokenizer,
            v_val_texts,
            cfg.i_max_seq_len,
            cfg.i_min_window_len,
            cfg.b_loss_mask_assistant_only,
            &cfg.s_assistant_marker,
        );

        if v_in.is_empty() || v_tgt.is_empty() {
            continue;
        }

        let a_input = Array2::from_shape_vec((1, v_in.len()), v_in.iter().map(|&id| id as f32).collect())
            .expect("invalid input shape");

        let mut a_forward = a_input;
        for layer in &mut llm.network {
            a_forward = layer.forward(&a_forward);
        }

        if a_forward.nrows() != v_tgt.len() {
            continue;
        }

        let a_probs = softmax_stable(&a_forward);
        d_loss_sum += masked_cross_entropy_loss_step(&a_probs, &v_tgt, &v_mask);

        let (hits, total) = masked_top1_hits(&a_probs, &v_tgt, &v_mask);
        i_top1_hits += hits;
        i_top1_total += total;

        i_steps += 1;
    }

    let d_loss = if i_steps > 0 { d_loss_sum / (i_steps as f32) } else { 0.0 };
    let d_acc = if i_top1_total > 0 {
        (i_top1_hits as f32) / (i_top1_total as f32)
    } else {
        0.0
    };

    (d_loss, d_acc)
}

fn masked_top1_hits(a_probs: &Array2<f32>, v_targets: &[usize], v_mask: &[u8]) -> (usize, usize) {
    let i_seq = a_probs.nrows();
    let i_vocab = a_probs.ncols();
    if i_seq == 0 || i_vocab == 0 {
        return (0, 0);
    }

    let i_tgt_len = v_targets.len().min(i_seq);
    let mut hits: usize = 0;
    let mut total: usize = 0;

    for i in 0..i_tgt_len {
        if i >= v_mask.len() || v_mask[i] == 0 {
            continue;
        }

        let mut best_j: usize = 0;
        let mut best_v: f32 = f32::NEG_INFINITY;

        for (j, &p) in a_probs.row(i).iter().enumerate() {
            if p > best_v {
                best_v = p;
                best_j = j;
            }
        }

        if best_j == v_targets[i] {
            hits += 1;
        }
        total += 1;
    }

    (hits, total)
}

// ---------------------------------------------------------------------------
// Train/val split by group to reduce leakage
// ---------------------------------------------------------------------------

fn split_train_val_by_prefix_group(v_texts: &[String], d_val_ratio: f32) -> (Vec<String>, Vec<String>) {
    // Group based split to reduce leakage. Supports:
    // - Explicit "[src=...]" prefix
    // - Fallback: stable group id derived from content prefix (hash)
    let d = d_val_ratio.clamp(0.0, 0.5);
    let n = v_texts.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut v_groups: Vec<String> = Vec::with_capacity(n);
    for s in v_texts {
        v_groups.push(extract_group_id(s));
    }

    let mut v_unique = v_groups.clone();
    v_unique.sort();
    v_unique.dedup();

    // Ensure at least 1 group in val when ratio > 0 and enough groups exist.
    let mut i_val_groups = ((v_unique.len() as f32) * d).round() as usize;
    if d > 0.0 && v_unique.len() >= 2 && i_val_groups == 0 {
        i_val_groups = 1;
    }
    i_val_groups = i_val_groups.min(v_unique.len());

    let mut set_val: std::collections::HashSet<String> = std::collections::HashSet::new();
    for g in v_unique.into_iter().take(i_val_groups) {
        set_val.insert(g);
    }

    let mut v_train: Vec<String> = Vec::new();
    let mut v_val: Vec<String> = Vec::new();

    for s in v_texts {
        let g = extract_group_id(s);
        // If explicit prefix exists, remove it before training, else keep text as-is.
        let s_payload = strip_group_prefix(s);
        if set_val.contains(&g) {
            v_val.push(s_payload);
        } else {
            v_train.push(s_payload);
        }
    }

    (v_train, v_val)
}

fn extract_group_id(s_in: &str) -> String {
    // Primary path: explicit prefix "[src=...]" at start.
    // Expected: "[src=ID]" then whitespace then payload.
    if let Some(s_id) = parse_src_prefix_id(s_in) {
        return format!("[src={}]", s_id);
    }

    // Fallback path: derive a stable group id from content.
    // This is a heuristic to reduce leakage when no explicit source id exists.
    // Strategy:
    // - take the first N bytes of normalized content
    // - hash with FNV-1a (stable, fast, no extra crates)
    // - bucketize to a moderate number of groups to enable splitting
    let s_norm = normalize_for_grouping(s_in);
    let i_bucket_count: u64 = 128;
    let i_hash = fnv1a_64(s_norm.as_bytes());
    let i_bucket = (i_hash % i_bucket_count) as u64;

    format!("[src=hash_bucket_{:03}]", i_bucket)
}

fn strip_group_prefix(s_in: &str) -> String {
    // Only strips if explicit "[src=...]" is present. Otherwise returns input.
    if let Some(pos) = s_in.find("]") {
        if s_in.starts_with("[src=") && pos + 1 <= s_in.len() {
            let rest = &s_in[(pos + 1)..];
            return rest.trim_start().to_string();
        }
    }
    s_in.to_string()
}

fn parse_src_prefix_id(s_in: &str) -> Option<String> {
    // Accepts "[src=...]" strictly at the beginning.
    if !s_in.starts_with("[src=") {
        return None;
    }
    let i_end = s_in.find("]")?;
    if i_end < 6 {
        return None;
    }
    // Extract between "[src=" and "]"
    let s_mid = &s_in[5..i_end];
    let s_mid = s_mid.trim();
    if s_mid.is_empty() {
        return None;
    }
    // Conservative ASCII-only id requirement.
    if !s_mid.is_ascii() {
        return None;
    }
    Some(s_mid.to_string())
}

fn normalize_for_grouping(s_in: &str) -> String {
    // Conservative normalization for hashing:
    // - trim
    // - collapse whitespace to single spaces
    // - take a bounded prefix length to reduce cost and reduce sensitivity
    let s_trim = s_in.trim();
    if s_trim.is_empty() {
        return String::new();
    }

    let mut out: Vec<u8> = Vec::with_capacity(256);
    let mut b_prev_space = false;

    for &b in s_trim.as_bytes().iter().take(512) {
        // drop ASCII control except space
        if b < 0x20 || b == 0x7F {
            continue;
        }
        if b == b' ' {
            if b_prev_space {
                continue;
            }
            b_prev_space = true;
            out.push(b' ');
            continue;
        }
        b_prev_space = false;
        out.push(b);
        if out.len() >= 256 {
            break;
        }
    }

    String::from_utf8_lossy(&out).to_string()
}

fn fnv1a_64(v: &[u8]) -> u64 {
    // FNV-1a 64-bit
    // History:
    //  - 2026-01-18: Added for stable fallback grouping without extra crates.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in v {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Window sampling with assistant only mask
// ---------------------------------------------------------------------------

fn sample_random_window_pair_from_texts_masked(
    tokenizer: &Tokenizer,
    v_texts: &[String],
    i_max_seq_len_requested: usize,
    i_min_window_len: usize,
    b_mask_assistant_only: bool,
    s_assistant_marker: &str,
) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
    use rand::Rng;

    if v_texts.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut rng = rand::thread_rng();
    let i_idx = rng.gen_range(0..v_texts.len());
    let s_raw = &v_texts[i_idx];

    let v_seq = tokenizer.encode_text(s_raw);
    if v_seq.len() < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let i_max_seq_len = i_max_seq_len_requested.min(MAX_SEQ_LEN_CANONICAL).max(2);
    let i_max_len_eff = i_max_seq_len.min(v_seq.len());
    let i_min_len_eff = i_min_window_len.clamp(2, i_max_len_eff);

    let i_win_len = if i_min_len_eff >= i_max_len_eff {
        i_max_len_eff
    } else {
        rng.gen_range(i_min_len_eff..=i_max_len_eff)
    };

    let i_start_max = v_seq.len().saturating_sub(i_win_len);
    let i_start = if i_start_max == 0 { 0 } else { rng.gen_range(0..=i_start_max) };

    let i_end = i_start + i_win_len;
    if i_end > v_seq.len() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let v_window = &v_seq[i_start..i_end];
    if v_window.len() < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let v_in = v_window[..v_window.len() - 1].to_vec();
    let v_tgt = v_window[1..].to_vec();

    let v_mask = if b_mask_assistant_only {
        build_assistant_only_mask(tokenizer, &v_in, &v_tgt, s_assistant_marker)
    } else {
        vec![1u8; v_tgt.len()]
    };

    (v_in, v_tgt, v_mask)
}

fn build_assistant_only_mask(
    tokenizer: &Tokenizer,
    v_in: &[usize],
    v_tgt: &[usize],
    s_assistant_marker: &str,
) -> Vec<u8> {
    // Strategy:
    // - decode v_in to text
    // - find last occurrence of assistant marker
    // - if absent, fallback to unmasked training to avoid zero-loss regression

    let s_in_text = tokenizer.decode_tokens(v_in);

    // Fallback: if marker not found, do NOT mask anything
    // This keeps Phase A (pretraining) learning even when no chat markers exist.
    let Some(i_pos) = s_in_text.rfind(s_assistant_marker) else {
        return vec![1u8; v_tgt.len()];
    };

    let mut v_mask = vec![0u8; v_tgt.len()];

    let i_cut = i_pos.saturating_add(s_assistant_marker.len());
    if i_cut > s_in_text.len() {
        return vec![1u8; v_tgt.len()];
    }

    let s_prefix = &s_in_text[..i_cut];

    // Encode prefix without EOS to approximate token boundary index.
    let v_prefix_tok = tokenizer.encode_stop_sequence(s_prefix);
    let i_prefix_len = v_prefix_tok.len();

    // v_tgt aligns with v_in shifted by 1
    let i_begin = i_prefix_len.saturating_sub(1);

    for i in 0..v_mask.len() {
        if i >= i_begin {
            v_mask[i] = 1;
        }
    }

    // Secondary safety: if still fully masked for any reason, unmask all
    if v_mask.iter().all(|&m| m == 0) {
        return vec![1u8; v_tgt.len()];
    }

    v_mask
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

fn argmax_row0(a_probs_1xv: &Array2<f32>) -> usize {
    if a_probs_1xv.nrows() != 1 || a_probs_1xv.ncols() == 0 {
        return 0;
    }

    let mut best_i: usize = 0;
    let mut best_v: f32 = f32::NEG_INFINITY;

    for (i, &p) in a_probs_1xv.row(0).iter().enumerate() {
        if p > best_v {
            best_v = p;
            best_i = i;
        }
    }

    best_i
}

fn apply_temperature_inplace(a_probs_1xv: &mut Array2<f32>, d_temperature: f32) {
    let d_temp = d_temperature.clamp(0.05, 5.0);
    if (d_temp - 1.0).abs() <= 1e-6 {
        return;
    }

    let d_inv = 1.0 / d_temp;
    let mut d_sum: f32 = 0.0;

    for p in a_probs_1xv.iter_mut() {
        *p = (*p).max(1e-20).powf(d_inv);
        d_sum += *p;
    }

    if d_sum > 0.0 {
        for p in a_probs_1xv.iter_mut() {
            *p /= d_sum;
        }
    }
}

fn sample_k_top_p_top(a_probs_1xv: &Array2<f32>, i_k_top: usize, d_p_top: f32) -> usize {
    use rand::Rng;

    if a_probs_1xv.nrows() != 1 {
        return 0;
    }
    let i_vocab = a_probs_1xv.ncols();
    if i_vocab == 0 {
        return 0;
    }

    let mut v_pairs: Vec<(usize, f32)> = a_probs_1xv
        .row(0)
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p.max(0.0)))
        .collect();

    v_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut v_filtered: Vec<(usize, f32)> = if i_k_top == 0 {
        v_pairs
    } else {
        let k = i_k_top.clamp(1, i_vocab);
        v_pairs.into_iter().take(k).collect()
    };

    let d_p = d_p_top.clamp(0.0, 1.0);
    if d_p <= 0.0 {
        return v_filtered.get(0).map(|x| x.0).unwrap_or(0);
    }

    if d_p < 1.0 {
        let mut d_cum: f32 = 0.0;
        let mut v_nucleus: Vec<(usize, f32)> = Vec::new();

        for &(i, p) in v_filtered.iter() {
            if p <= 0.0 {
                continue;
            }
            v_nucleus.push((i, p));
            d_cum += p;
            if d_cum >= d_p && !v_nucleus.is_empty() {
                break;
            }
        }

        if !v_nucleus.is_empty() {
            v_filtered = v_nucleus;
        }
    }

    let mut sum: f32 = 0.0;
    for &(_, p) in v_filtered.iter() {
        sum += p;
    }

    if sum <= 0.0 {
        return v_filtered.get(0).map(|x| x.0).unwrap_or(0);
    }

    let mut rng = rand::thread_rng();
    let mut r = rng.gen_range(0.0..sum);

    for (i, p) in v_filtered {
        if r <= p {
            return i;
        }
        r -= p;
    }

    0
}

// ---------------------------------------------------------------------------
// Stop sequence helpers
// ---------------------------------------------------------------------------

fn ends_with_any(v_out: &[usize], v_stop: &[Vec<usize>]) -> bool {
    for s in v_stop {
        if s.is_empty() {
            continue;
        }
        if v_out.len() < s.len() {
            continue;
        }
        let a = &v_out[v_out.len() - s.len()..];
        if a == s.as_slice() {
            return true;
        }
    }
    false
}
