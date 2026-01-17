// math.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Numeric utilities: AdamW, stable softmax, mixed precision helpers,
//           and gradient scaling.
// History:
//  - 2026-01-17: Add AdamW, GradScaler, stable softmax (log-sum-exp based),
//                and mixed precision conversion helpers.
// ============================================================================

#![forbid(unsafe_code)]

use bincode::{Decode, Encode};
use ndarray::{Array2, Axis, Zip};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub use half::{bf16, f16};

use rand::Rng;

// ---------------- Optimizer: AdamW (with accumulation) ----------------

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct AdamW {
    #[bincode(with_serde)]
    m: Array2<f32>,
    #[bincode(with_serde)]
    v: Array2<f32>,
    t: usize,

    i_accumulate: usize,
    i_since_update: usize,
    #[bincode(with_serde)]
    grad_buf: Array2<f32>,

    beta1: f32,
    beta2: f32,
    eps: f32,

    // AdamW decoupled weight decay
    wd: f32,
}

impl AdamW {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            t: 0,
            i_accumulate: 1,
            i_since_update: 0,
            grad_buf: Array2::zeros(shape),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd: 0.01,
        }
    }

    pub fn set_accumulate_steps(&mut self, i_steps: usize) {
        self.i_accumulate = i_steps.max(1);
        self.i_since_update = 0;
        self.grad_buf.fill(0.0);
    }

    pub fn set_weight_decay(&mut self, d_wd: f32) {
        self.wd = d_wd.max(0.0);
    }

    // AdamW: weight decay is decoupled from gradient term:
    // w = w - lr * (adam_update) - lr * wd * w
    pub fn step(&mut self, w: &mut Array2<f32>, grad: &Array2<f32>, d_lr: f32) {
        if !d_lr.is_finite() || d_lr <= 0.0 {
            return;
        }

        Zip::from(&mut self.grad_buf)
            .and(grad)
            .for_each(|gb, &g| *gb += g);

        self.i_since_update += 1;
        if self.i_since_update < self.i_accumulate {
            return;
        }

        let d_scale = 1.0 / (self.i_accumulate as f32);
        let mut g_avg = self.grad_buf.clone();
        g_avg.mapv_inplace(|x| x * d_scale);

        self.t += 1;
        let t = self.t as f32;

        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;

        Zip::from(&mut self.m)
            .and(&g_avg)
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g);

        Zip::from(&mut self.v)
            .and(&g_avg)
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * g * g);

        let bias_c1 = 1.0 - b1.powf(t);
        let bias_c2 = 1.0 - b2.powf(t);

        let d_wd = self.wd;
        for ((w_ij, m_ij), v_ij) in w.iter_mut().zip(self.m.iter()).zip(self.v.iter()) {
            let m_hat = *m_ij / bias_c1;
            let v_hat = *v_ij / bias_c2;
            let adam = m_hat / (v_hat.sqrt() + eps);
            *w_ij -= d_lr * adam;
            *w_ij -= d_lr * d_wd * *w_ij;
        }

        self.grad_buf.fill(0.0);
        self.i_since_update = 0;
    }
}

// ---------------- GradScaler for mixed precision style training ----------------
// Note: This implementation scales loss-equivalent gradients at the last stage.
// It also performs basic overflow detection and dynamic scaling.

#[derive(Clone, Debug)]
pub struct GradScaler {
    d_scale: f32,
    d_growth: f32,
    d_backoff: f32,
    i_growth_interval: usize,
    i_good_steps: usize,
    d_min_scale: f32,
    d_max_scale: f32,
}

impl GradScaler {
    pub fn new_default() -> Self {
        Self {
            d_scale: 1024.0,
            d_growth: 2.0,
            d_backoff: 0.5,
            i_growth_interval: 200,
            i_good_steps: 0,
            d_min_scale: 1.0,
            d_max_scale: 65536.0,
        }
    }

    pub fn scale(&self) -> f32 {
        self.d_scale
    }

    pub fn scale_grads_inplace(&self, a_grads: &mut Array2<f32>) {
        let d = self.d_scale;
        if !d.is_finite() || d <= 0.0 {
            return;
        }
        a_grads.mapv_inplace(|x| x * d);
    }

    pub fn unscale_grads_inplace(&self, a_grads: &mut Array2<f32>) {
        let d = self.d_scale;
        if !d.is_finite() || d <= 0.0 {
            return;
        }
        let inv = 1.0 / d;
        a_grads.mapv_inplace(|x| x * inv);
    }

    pub fn grads_are_finite(a_grads: &Array2<f32>) -> bool {
        a_grads.iter().all(|v| v.is_finite())
    }

    pub fn update(&mut self, b_overflow: bool) {
        if b_overflow {
            self.d_scale = (self.d_scale * self.d_backoff).clamp(self.d_min_scale, self.d_max_scale);
            self.i_good_steps = 0;
        } else {
            self.i_good_steps += 1;
            if self.i_good_steps >= self.i_growth_interval {
                self.d_scale = (self.d_scale * self.d_growth).clamp(self.d_min_scale, self.d_max_scale);
                self.i_good_steps = 0;
            }
        }
    }
}

// ---------------- Softmax: stable log-sum-exp implementation ----------------

pub fn softmax_stable(a_logits: &Array2<f32>) -> Array2<f32> {
    let mut out = a_logits.clone();
    for mut row in out.rows_mut() {
        let d_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut d_sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - d_max).exp();
            d_sum += *v;
        }
        if d_sum > 0.0 && d_sum.is_finite() {
            let inv = 1.0 / d_sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        } else {
            // Defensive fallback: uniform
            let n = row.len().max(1) as f32;
            for v in row.iter_mut() {
                *v = 1.0 / n;
            }
        }
    }
    out
}

pub fn softmax_stable_rows_par(m: &Array2<f32>) -> Array2<f32> {
    let mut out = m.clone();
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let d_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut d_sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - d_max).exp();
                d_sum += *v;
            }
            if d_sum > 0.0 && d_sum.is_finite() {
                let inv = 1.0 / d_sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            } else {
                let n = row.len().max(1) as f32;
                for v in row.iter_mut() {
                    *v = 1.0 / n;
                }
            }
        });
    out
}

pub fn softmax_backward_rows(softmax_out: &Array2<f32>, grad_out: &Array2<f32>) -> Array2<f32> {
    let mut grad_in = Array2::<f32>::zeros(softmax_out.dim());
    for ((mut g_row, s_row), go_row) in grad_in
        .axis_iter_mut(Axis(0))
        .zip(softmax_out.axis_iter(Axis(0)))
        .zip(grad_out.axis_iter(Axis(0)))
    {
        let dot: f32 = s_row
            .iter()
            .zip(go_row.iter())
            .map(|(&y, &dy)| y * dy)
            .sum();
        for ((g, &y), &dy) in g_row.iter_mut().zip(s_row.iter()).zip(go_row.iter()) {
            *g = y * (dy - dot);
        }
    }
    grad_in
}

// ---------------- Loss and gradient helpers (unchanged API) ----------------

pub fn cross_entropy_loss_step(a_probs: &Array2<f32>, v_target: &[usize]) -> f32 {
    if v_target.is_empty() {
        return 0.0;
    }
    let mut d_loss = 0.0;
    for (i_row, &i_tgt) in v_target.iter().enumerate() {
        if i_row >= a_probs.nrows() || i_tgt >= a_probs.ncols() {
            continue;
        }
        let d_p = a_probs[(i_row, i_tgt)].max(1e-15);
        d_loss -= d_p.ln();
    }
    d_loss / (v_target.len() as f32)
}

pub fn compute_gradients_step(a_probs: &Array2<f32>, v_target: &[usize]) -> Array2<f32> {
    let mut a_grad = a_probs.clone();
    if v_target.is_empty() {
        return a_grad;
    }
    for (i_row, &i_tgt) in v_target.iter().enumerate() {
        if i_row < a_grad.nrows() && i_tgt < a_grad.ncols() {
            a_grad[(i_row, i_tgt)] -= 1.0;
        }
    }
    let d_batch = v_target.len() as f32;
    a_grad.mapv(|x| x / d_batch.max(1.0))
}

pub fn clip_gradients(a_grads: &mut Array2<f32>, d_max_norm: f32) {
    if !d_max_norm.is_finite() || d_max_norm <= 0.0 {
        return;
    }
    let d_norm: f32 = a_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if d_norm.is_finite() && d_norm > d_max_norm {
        let d_scale = d_max_norm / d_norm;
        a_grads.mapv_inplace(|x| x * d_scale);
    }
}

// ---------------- Dropout helper (keep for other code paths) ----------------

pub fn dropout_inplace(m_tensor: &mut Array2<f32>, f_rate: f32) {
    if f_rate <= 0.0 {
        return;
    }
    let p_drop = f_rate.clamp(0.0, 1.0);
    let scale = if p_drop < 1.0 { 1.0 / (1.0 - p_drop) } else { 0.0 };
    let mut rng = rand::rng();
    for elem in m_tensor.iter_mut() {
        let drop_now: bool = rng.random::<f32>() < p_drop;
        if drop_now {
            *elem = 0.0;
        } else {
            *elem *= scale;
        }
    }
}

// ---------------- RoPE and mixed precision conversions (unchanged) ----------------

use ndarray::{ArrayViewMut2, Axis as Ax};

pub fn apply_rope(mut q: ArrayViewMut2<'_, f32>, mut k: ArrayViewMut2<'_, f32>, i_step: usize, i_head_dim: usize) {
    let (seq_q, dim_q) = q.dim();
    let (seq_k, dim_k) = k.dim();
    assert_eq!(dim_q, dim_k, "q and k dim mismatch");
    assert_eq!(seq_q, seq_k, "q and k seq mismatch");
    assert!(i_head_dim > 0 && i_head_dim % 2 == 0, "head_dim invalid");
    assert!(dim_q % i_head_dim == 0, "embed not multiple of head_dim");

    let i_heads = dim_q / i_head_dim;
    let half = i_head_dim / 2;
    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2.0 * i as f32) / i_head_dim as f32;
        inv_freq.push(10000_f32.powf(-exp));
    }
    for (i_row, (mut vq_row, mut vk_row)) in q.axis_iter_mut(Ax(0)).zip(k.axis_iter_mut(Ax(0))).enumerate() {
        let pos = (i_step + i_row) as f32;
        for h in 0..i_heads {
            let base = h * i_head_dim;
            for i in 0..half {
                let theta = pos * inv_freq[i];
                let (s, c) = theta.sin_cos();
                let i0 = base + 2 * i;
                let i1 = base + 2 * i + 1;
                let q0 = vq_row[i0];
                let q1 = vq_row[i1];
                vq_row[i0] = q0 * c - q1 * s;
                vq_row[i1] = q0 * s + q1 * c;

                let k0 = vk_row[i0];
                let k1 = vk_row[i1];
                vk_row[i0] = k0 * c - k1 * s;
                vk_row[i1] = k0 * s + k1 * c;
            }
        }
    }
}

pub fn apply_rope_backward(mut dq: ArrayViewMut2<'_, f32>, mut dk: ArrayViewMut2<'_, f32>, i_step: usize, i_head_dim: usize) {
    let (seq_q, dim_q) = dq.dim();
    let (seq_k, dim_k) = dk.dim();
    assert_eq!(dim_q, dim_k, "dq and dk dim mismatch");
    assert_eq!(seq_q, seq_k, "dq and dk seq mismatch");
    assert!(i_head_dim > 0 && i_head_dim % 2 == 0, "head_dim invalid");
    assert!(dim_q % i_head_dim == 0, "embed not multiple of head_dim");

    let i_heads = dim_q / i_head_dim;
    let half = i_head_dim / 2;
    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2.0 * i as f32) / i_head_dim as f32;
        inv_freq.push(10000_f32.powf(-exp));
    }
    for (i_row, (mut vq_row, mut vk_row)) in dq.axis_iter_mut(Ax(0)).zip(dk.axis_iter_mut(Ax(0))).enumerate() {
        let pos = (i_step + i_row) as f32;
        for h in 0..i_heads {
            let base = h * i_head_dim;
            for i in 0..half {
                let theta = pos * inv_freq[i];
                let (s, c) = (-theta).sin_cos();
                let i0 = base + 2 * i;
                let i1 = base + 2 * i + 1;
                let q0 = vq_row[i0];
                let q1 = vq_row[i1];
                vq_row[i0] = q0 * c - q1 * s;
                vq_row[i1] = q0 * s + q1 * c;

                let k0 = vk_row[i0];
                let k1 = vk_row[i1];
                vk_row[i0] = k0 * c - k1 * s;
                vk_row[i1] = k0 * s + k1 * c;
            }
        }
    }
}

pub fn to_f16(m: &Array2<f32>) -> ndarray::Array2<f16> {
    m.mapv(f16::from_f32)
}
pub fn from_f16(m: &ndarray::Array2<f16>) -> Array2<f32> {
    m.mapv(|h| f32::from(h))
}
pub fn to_bf16(m: &Array2<f32>) -> ndarray::Array2<bf16> {
    m.mapv(bf16::from_f32)
}
pub fn from_bf16(m: &ndarray::Array2<bf16>) -> Array2<f32> {
    m.mapv(|h| f32::from(h))
}
