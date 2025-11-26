/***********************************************************************
*  layer_pos_encoding.rs  –  Rotary Relative Position Embedding (RoPE)
***********************************************************************/
use ndarray::{ArrayViewMut2, Axis};

/// Wendet RoPE in-place auf Query und Key an (pro Head).
/// Erwartet: dim % head_dim == 0, head_dim % 2 == 0.
pub fn apply_rope(
    mut q: ArrayViewMut2<'_, f32>,   // [seq, embed]
    mut k: ArrayViewMut2<'_, f32>,   // [seq, embed]
    i_step: usize,
    i_head_dim: usize,               // Größe pro Head
) {
    let (seq_q, dim_q) = q.dim();
    let (seq_k, dim_k) = k.dim();

    // Grundchecks
    assert_eq!(dim_q, dim_k, "q und k haben unterschiedliche Embedding-Dimension");
    assert_eq!(seq_q, seq_k, "q und k haben unterschiedliche Sequenzlänge");
    assert!(i_head_dim > 0, "head_dim muss > 0 sein");
    assert!(i_head_dim % 2 == 0, "head_dim muss gerade sein");
    assert!(dim_q % i_head_dim == 0, "embed muss ein Vielfaches von head_dim sein");

    let i_heads = dim_q / i_head_dim;
    let half = i_head_dim / 2;

    // Frequenzen vorab berechnen: inv_freq[i] = 10000^(-2i / head_dim)
    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2.0 * i as f32) / i_head_dim as f32;
        inv_freq.push(10000_f32.powf(-exp));
    }

    for (i_row, (mut vq_row, mut vk_row)) in
        q.axis_iter_mut(Axis(0)).zip(k.axis_iter_mut(Axis(0))).enumerate()
    {
        let pos = (i_step + i_row) as f32;

        // pro Head blockweise rotieren
        for h in 0..i_heads {
            let base = h * i_head_dim;
            for i in 0..half {
                let theta = pos * inv_freq[i];
                let (s, c) = theta.sin_cos();

                // Index-Paar innerhalb des Heads: (2i, 2i+1)
                let i0 = base + 2 * i;
                let i1 = base + 2 * i + 1;

                // Query rotieren
                let q0 = vq_row[i0];
                let q1 = vq_row[i1];
                vq_row[i0] = q0 * c - q1 * s;
                vq_row[i1] = q0 * s + q1 * c;

                // Key rotieren
                let k0 = vk_row[i0];
                let k1 = vk_row[i1];
                vk_row[i0] = k0 * c - k1 * s;
                vk_row[i1] = k0 * s + k1 * c;
            }
        }
    }
}
