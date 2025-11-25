/***********************************************************************
*  layer_pos_encoding.rs  –  Rotary Relative Position Embedding (RoPE)
***********************************************************************/
use ndarray::{ArrayViewMut2, Axis};

/// wendet RoPE in-place auf Query und Key an
pub fn apply_rope(
    mut q: ArrayViewMut2<'_, f32>,
    mut k: ArrayViewMut2<'_, f32>,
    i_step: usize,
) {
    let (seq_q, dim_q) = q.dim();
    let (seq_k, dim_k) = k.dim();

    assert_eq!(dim_q, dim_k, "q und k haben unterschiedliche Embedding-Dimension");
    assert_eq!(seq_q, seq_k, "q und k haben unterschiedliche Sequenzlänge");
    assert!(dim_q % 2 == 0, "Embedding-Dimension muss gerade sein");

    let half = dim_q / 2;

    for (i_row, (mut vq_row, mut vk_row)) in
        q.axis_iter_mut(Axis(0)).zip(k.axis_iter_mut(Axis(0))).enumerate()
    {
        let pos = i_step + i_row; // jede Zeile bekommt ihre eigene Position
        for j in 0..half {
            let exp = (2.0 * j as f32) / dim_q as f32;
            // klassisches RoPE: kein PI-Faktor
            let theta = (pos as f32) / 10000_f32.powf(exp);
            let (sin, cos) = theta.sin_cos();

            // Query rotieren
            let q1 = vq_row[j];
            let q2 = vq_row[j + half];
            vq_row[j]        =  q1 * cos - q2 * sin;
            vq_row[j + half] =  q1 * sin + q2 * cos;

            // Key rotieren
            let k1 = vk_row[j];
            let k2 = vk_row[j + half];
            vk_row[j]        =  k1 * cos - k2 * sin;
            vk_row[j + half] =  k1 * sin + k2 * cos;
        }
    }
}
