// src/feed_forward_geglu.rs
// FeedForwardGeGLU – jetzt mit Adam, Backward und Dropout-Masken
use crate::adam::Adam;
use crate::utils::apply_weight_decay;
use bincode::{Decode, Encode};
use ndarray::{Array2, Axis, Zip};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct FeedForwardGeGLU {
    #[bincode(with_serde)]
    pub w_in: Array2<f32>, // [embed, 2*hidden]
    #[bincode(with_serde)]
    pub w_out: Array2<f32>, // [hidden, embed]
    pub f_dropout: f32,
    pub f_decay: f32,
    // Optimizer
    #[bincode(with_serde)]
    optimizer_w_in: Adam,
    #[bincode(with_serde)]
    optimizer_w_out: Adam,
    // Caches (nur Runtime)
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_lin: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_gate: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_gate_gelu: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_act: Option<Array2<f32>>,
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_dropout_mask: Option<Array2<f32>>,
}

impl FeedForwardGeGLU {
    pub fn new(i_embed: usize, i_hidden: usize, f_dropout: f32) -> Self {
        assert!(i_embed > 0 && i_hidden > 0, "Dimensionen muessen > 0 sein");

        let mut rng = rand::thread_rng();
        let std_in = (2.0 / i_embed as f32).sqrt();
        let std_out = (2.0 / i_hidden as f32).sqrt();
        let normal_in = Normal::new(0.0, std_in).unwrap();
        let normal_out = Normal::new(0.0, std_out).unwrap();

        FeedForwardGeGLU {
            w_in: ndarray::Array2::from_shape_fn((i_embed, i_hidden * 2), |_| {
                normal_in.sample(&mut rng)
            }),
            w_out: ndarray::Array2::from_shape_fn((i_hidden, i_embed), |_| {
                normal_out.sample(&mut rng)
            }),
            f_dropout,
            f_decay: 1e-4,
            optimizer_w_in: Adam::new((i_embed, i_hidden * 2)),
            optimizer_w_out: Adam::new((i_hidden, i_embed)),
            cached_input: None,
            cached_lin: None,
            cached_gate: None,
            cached_gate_gelu: None,
            cached_act: None,
            cached_dropout_mask: None,
        }
    }

    pub fn set_accumulate_steps(&mut self, steps: usize) {
        self.optimizer_w_in.set_accumulate_steps(steps);
        self.optimizer_w_out.set_accumulate_steps(steps);
    }

    pub fn parameter_count(&self) -> usize {
        self.w_in.len() + self.w_out.len()
    }

    // internes Dropout mit Masken (inverted dropout)
    fn apply_dropout(&self, m: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
        let p = self.f_dropout.clamp(0.0, 1.0);
        if p <= 0.0 {
            return (m.clone(), None);
        }
        let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
        let mut rng = rand::thread_rng();
        let mut out = m.clone();
        let mut mask = m.clone();
        for (o, mk) in out.iter_mut().zip(mask.iter_mut()) {
            let drop = rng.r#gen::<f32>() < p;
            if drop {
                *o = 0.0;
                *mk = 0.0;
            } else {
                *o *= scale;
                *mk = scale;
            }
        }
        (out, Some(mask))
    }

    pub fn forward(&mut self, m_x: &Array2<f32>) -> Array2<f32> {
        // Projizieren und splitten
        let m_proj = m_x.dot(&self.w_in); // [seq, 2h]
        let cols = m_proj.dim().1;
        assert!(cols % 2 == 0, "Proj-Spaltenzahl muss durch 2 teilbar sein");
        let (m_lin_v, m_gate_v) = m_proj.view().split_at(Axis(1), cols / 2);
        let m_lin = m_lin_v.to_owned(); // [seq, h]
        let m_gate = m_gate_v.to_owned(); // [seq, h]
        let m_gate_gelu = m_gate.mapv(gelu); // [seq, h]
        let m_act = &m_lin * &m_gate_gelu; // [seq, h]
        let m_out_raw = m_act.dot(&self.w_out); // [seq, embed]
        let (m_out, mask) = self.apply_dropout(&m_out_raw);

        // Weight Decay (L2 als shrink)
        apply_weight_decay(&mut self.w_in, self.f_decay);
        apply_weight_decay(&mut self.w_out, self.f_decay);

        // Caches
        self.cached_input = Some(m_x.clone());
        self.cached_lin = Some(m_lin);
        self.cached_gate = Some(m_gate.clone());
        self.cached_gate_gelu = Some(m_gate_gelu);
        self.cached_act = Some(m_act);
        self.cached_dropout_mask = mask;

        m_out
    }

    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Caches
        let m_x = self.cached_input.as_ref().expect("forward zuerst");
        let m_lin = self.cached_lin.as_ref().expect("lin fehlt");
        let m_gate = self.cached_gate.as_ref().expect("gate fehlt");
        let m_gate_gelu = self.cached_gate_gelu.as_ref().expect("gate_gelu fehlt");
        let m_act = self.cached_act.as_ref().expect("act fehlt");
        // Dropout-Maske anwenden
        let mut grad_after_dropout = grads.clone();
        if let Some(mask) = self.cached_dropout_mask.as_ref() {
            Zip::from(&mut grad_after_dropout)
                .and(mask)
                .for_each(|g, &m| *g = *g * m);
        }

        // dL/dW_out und dL/dact
        let grad_w_out = m_act.t().dot(&grad_after_dropout); // [h, embed]
        let grad_act = grad_after_dropout.dot(&self.w_out.t()); // [seq, h]

        // act = lin * gelu(gate)
        let grad_lin = &grad_act * m_gate_gelu; // [seq, h]
        let grad_gate_gelu = &grad_act * m_lin; // [seq, h]
        let grad_gate = grad_gate_gelu.mapv(dgelu); // [seq, h]

        // zusammenführen für w_in
        let (seq, h) = (grad_lin.dim().0, grad_lin.dim().1);
        let mut grad_proj = Array2::<f32>::zeros((seq, 2 * h));
        {
            let mut left = grad_proj.slice_mut(ndarray::s![.., 0..h]);
            left.assign(&grad_lin);
        }
        {
            let mut right = grad_proj.slice_mut(ndarray::s![.., h..2 * h]);
            right.assign(&grad_gate);
        }

        let grad_w_in = m_x.t().dot(&grad_proj); // [embed, 2h]
        let grad_x = grad_proj.dot(&self.w_in.t()); // [seq, embed]

        // Optimizer-Schritt (mit Akkumulation)
        self.optimizer_w_out.step(&mut self.w_out, &grad_w_out, lr);
        self.optimizer_w_in.step(&mut self.w_in, &grad_w_in, lr);

        grad_x
    }
}

fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

// Ableitung der tanh-GELU-Approximation
fn dgelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    let x3 = x * x * x;
    let t = SQRT_2_OVER_PI * (x + 0.044715 * x3);
    let th = t.tanh();
    let sech2 = 1.0 - th * th;
    0.5 * (1.0 + th) + 0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
}
