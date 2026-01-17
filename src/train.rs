// train.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Integrates GradScaler and stable softmax, plus train/eval mode gating
//           for dropout and stochastic depth.
// History:
//  - 2026-01-17: Add gradient scaling and train/eval propagation.
// ============================================================================

#![forbid(unsafe_code)]

use anyhow::{anyhow, Context, Result};
use bincode::{config, decode_from_std_read, encode_into_std_write};
use ndarray::{Array2, Axis};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use crate::layers::{Embeddings, Layer, OutputProjection, TransformerBlockV2};
use crate::math::{clip_gradients, compute_gradients_step, cross_entropy_loss_step, softmax_stable, GradScaler};
use crate::tokenize::Tokenizer;
use crate::utils::MAX_SEQ_LEN_CANONICAL;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub d_temperature: f32,
    pub i_k_top: usize,
    pub d_p_top: f32,
    pub b_greedy: bool,
}

impl SamplingParams {
    pub fn sane_default() -> Self {
        Self {
            d_temperature: 0.9,
            i_k_top: 50,
            d_p_top: 0.95,
            b_greedy: false,
        }
    }

    pub fn greedy_debug() -> Self {
        Self {
            d_temperature: 1.0,
            i_k_top: 0,
            d_p_top: 0.0,
            b_greedy: true,
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
    }
}

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
        // Inference must run without dropout and stochastic depth.
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
                        };

                        println!(
                            "SAMPLE: temp={:.2} k_top={} p_top={:.2}",
                            d_temp, i_k, d_p
                        );

                        let _ = self.generate(s_prompt, &params, true);
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

    pub fn predict(&mut self, s_input: &str) -> String {
        self.set_eval_mode_no_dropout();
        let params = SamplingParams::sane_default();
        let v_ids = self.generate(s_input, &params, true);
        self.tokenizer.decode_tokens(&v_ids)
    }

    pub fn predict_greedy(&mut self, s_input: &str) -> String {
        self.set_eval_mode_no_dropout();
        let params = SamplingParams::greedy_debug();
        let v_ids = self.generate(s_input, &params, true);
        self.tokenizer.decode_tokens(&v_ids)
    }

    fn generate(&mut self, s_input: &str, params: &SamplingParams, b_stream: bool) -> Vec<usize> {
        let mut v_context: Vec<usize> = self.tokenizer.encode_text(s_input);
        let i_eos = self.tokenizer.eos_id();

        let i_cap = self.i_max_seq_len_runtime.min(MAX_SEQ_LEN_CANONICAL);

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

            // keep existing sampling logic; uses stable softmax for logits -> probs
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

            if v_context.len() >= i_cap {
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

    pub fn train_two_phase(&mut self, v_pretrain_texts: &[String], v_main_texts: &[String], cfg_pretrain: &TrainConfig, cfg_main: &TrainConfig) {
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
        self.set_train_mode(true);

        let mut grad_scaler = GradScaler::new_default();

        let stop_flag = Arc::new(AtomicBool::new(false));
        {
            let stop_flag_ctrlc = Arc::clone(&stop_flag);
            let _ = ctrlc::set_handler(move || {
                stop_flag_ctrlc.store(true, AtomicOrdering::SeqCst);
            });
        }

        let (v_train_texts, v_val_texts) = split_train_val(v_texts, cfg.d_val_ratio);

        let v_train_tok: Vec<Vec<usize>> = v_train_texts.iter().map(|s| self.tokenizer.encode_text(s)).collect();
        let v_val_tok: Vec<Vec<usize>> = v_val_texts.iter().map(|s| self.tokenizer.encode_text(s)).collect();

        println!(
            "TRAIN: samples={} VAL: samples={} max_seq_len={} lr={:.8} steps_per_epoch={} batch_size={} attn_do={:.3} res_do={:.3} sd_p={:.3} grad_scaling={}",
            v_train_tok.len(),
            v_val_tok.len(),
            cfg.i_max_seq_len,
            cfg.d_lr,
            cfg.i_steps_per_epoch,
            cfg.i_batch_size,
            cfg.d_attention_dropout,
            cfg.d_residual_dropout,
            cfg.d_stochastic_depth_p,
            cfg.b_grad_scaling
        );

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

                let (v_in, v_tgt) = sample_random_window_pair(&v_train_tok, cfg.i_max_seq_len, cfg.i_min_window_len);
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
                d_loss_sum += cross_entropy_loss_step(&a_probs, &v_tgt);

                let mut a_grads = compute_gradients_step(&a_probs, &v_tgt);

                // Mixed precision style: scale grads before clipping and backward,
                // then unscale before optimizer step inside layers via AdamW.
                // Since optimizers are inside layers, we unscale before calling backward.
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

                for layer in self.network.iter_mut().rev() {
                    a_grads = layer.backward(&a_grads, cfg.d_lr);
                }

                i_steps += 1;
                i_tokens += v_tgt.len();

                if cfg.b_grad_scaling {
                    grad_scaler.update(false);
                }
            }

            self.set_eval_mode_no_dropout();
            let (d_val_loss, d_val_acc) = evaluate_validation(self, &v_val_tok, &cfg);

            let d_train_loss = if i_steps > 0 { d_loss_sum / (i_steps as f32) } else { 0.0 };
            let d_secs = t_start.elapsed().as_secs_f32().max(1e-6);
            let d_tps = (i_tokens as f32) / d_secs;

            println!(
                "Epoch {} train_loss {:.4} val_loss {:.4} val_acc {:.4} tokens_s {:.0} steps {}",
                i_epoch, d_train_loss, d_val_loss, d_val_acc, d_tps, i_steps
            );

            // restore train mode for next epoch
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

// ---------------- Validation helpers (use stable softmax) ----------------

fn evaluate_validation(llm: &mut LLM, v_val_tok: &[Vec<usize>], cfg: &TrainConfig) -> (f32, f32) {
    if v_val_tok.is_empty() {
        return (0.0, 0.0);
    }

    let mut d_loss_sum: f32 = 0.0;
    let mut i_steps: usize = 0;
    let mut i_top1_hits: usize = 0;
    let mut i_top1_total: usize = 0;

    let i_eval_steps = (cfg.i_steps_per_epoch / 4).max(50);

    for _ in 0..i_eval_steps {
        let (v_in, v_tgt) = sample_random_window_pair(v_val_tok, cfg.i_max_seq_len, cfg.i_min_window_len);
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
        d_loss_sum += cross_entropy_loss_step(&a_probs, &v_tgt);

        let (hits, total) = top1_hits(&a_probs, &v_tgt);
        i_top1_hits += hits;
        i_top1_total += total;

        i_steps += 1;
    }

    let d_loss = if i_steps > 0 { d_loss_sum / (i_steps as f32) } else { 0.0 };
    let d_acc = if i_top1_total > 0 { (i_top1_hits as f32) / (i_top1_total as f32) } else { 0.0 };
    (d_loss, d_acc)
}

fn top1_hits(a_probs: &Array2<f32>, v_targets: &[usize]) -> (usize, usize) {
    let i_seq = a_probs.nrows();
    let i_vocab = a_probs.ncols();
    if i_seq == 0 || i_vocab == 0 {
        return (0, 0);
    }

    let i_tgt_len = v_targets.len().min(i_seq);
    let mut hits: usize = 0;

    for i in 0..i_tgt_len {
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
    }

    (hits, i_tgt_len)
}

// ---------------- Sampling helpers (unchanged) ----------------

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

// ---------------- Train/val split and window sampling (unchanged) ----------------

fn split_train_val(v_texts: &[String], d_val_ratio: f32) -> (Vec<String>, Vec<String>) {
    let d = d_val_ratio.clamp(0.0, 0.5);
    let n = v_texts.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let i_val = ((n as f32) * d).round() as usize;
    let i_val = i_val.min(n);
    let i_train = n - i_val;

    (v_texts[..i_train].to_vec(), v_texts[i_train..].to_vec())
}

fn sample_random_window_pair(v_sequences: &[Vec<usize>], i_max_seq_len_requested: usize, i_min_window_len: usize) -> (Vec<usize>, Vec<usize>) {
    use rand::Rng;

    if v_sequences.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut rng = rand::thread_rng();
    let i_seq_idx = rng.gen_range(0..v_sequences.len());
    let v_seq = &v_sequences[i_seq_idx];

    if v_seq.len() < 2 {
        return (Vec::new(), Vec::new());
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
        return (Vec::new(), Vec::new());
    }

    let v_window = &v_seq[i_start..i_end];
    if v_window.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    let v_in = v_window[..v_window.len() - 1].to_vec();
    let v_tgt = v_window[1..].to_vec();
    (v_in, v_tgt)
}
