// src/layer_pos_encoding.rs
// RoPE: Vorwärts gibt es schon. Hier die inverse Rotation für Backward.
use ndarray::{ArrayViewMut2, Axis};

pub fn apply_rope(
    mut q: ArrayViewMut2<'_, f32>,
    mut k: ArrayViewMut2<'_, f32>,
    i_step: usize,
    i_head_dim: usize,
) {
    let (seq_q, dim_q) = q.dim();
    let (seq_k, dim_k) = k.dim();
    assert_eq!(dim_q, dim_k, "q und k haben unterschiedliche Embedding-Dimension");
    assert_eq!(seq_q, seq_k, "q und k haben unterschiedliche Sequenzlaengen");
    assert!(i_head_dim > 0 && i_head_dim % 2 == 0, "head_dim ungueltig");
    assert!(dim_q % i_head_dim == 0, "embed kein Vielfaches von head_dim");

    let i_heads = dim_q / i_head_dim;
    let half = i_head_dim / 2;
    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2.0 * i as f32) / i_head_dim as f32;
        inv_freq.push(10000_f32.powf(-exp));
    }
    for (i_row, (mut vq_row, mut vk_row)) in
        q.axis_iter_mut(Axis(0)).zip(k.axis_iter_mut(Axis(0))).enumerate()
    {
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

// inverse Rotation (für Backward-Pass)
pub fn apply_rope_backward(
    mut dq: ArrayViewMut2<'_, f32>,
    mut dk: ArrayViewMut2<'_, f32>,
    i_step: usize,
    i_head_dim: usize,
) {
    let (seq_q, dim_q) = dq.dim();
    let (seq_k, dim_k) = dk.dim();
    assert_eq!(dim_q, dim_k, "dq und dk haben unterschiedliche Embedding-Dimension");
    assert_eq!(seq_q, seq_k, "dq und dk haben unterschiedliche Sequenzlaengen");
    assert!(i_head_dim > 0 && i_head_dim % 2 == 0, "head_dim ungueltig");
    assert!(dim_q % i_head_dim == 0, "embed kein Vielfaches von head_dim");

    let i_heads = dim_q / i_head_dim;
    let half = i_head_dim / 2;
    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2.0 * i as f32) / i_head_dim as f32;
        inv_freq.push(10000_f32.powf(-exp));
    }
    for (i_row, (mut vq_row, mut vk_row)) in
        dq.axis_iter_mut(Axis(0)).zip(dk.axis_iter_mut(Axis(0))).enumerate()
    {
        let pos = (i_step + i_row) as f32;
        for h in 0..i_heads {
            let base = h * i_head_dim;
            for i in 0..half {
                let theta = pos * inv_freq[i];
                let (s, c) = (-theta).sin_cos(); // inverse Rotation
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
