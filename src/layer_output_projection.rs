// ===========================================================================
//  Datei:        layer_output_projection.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        Output Projection Layer
//  Autor:        Marcus Schlieper (ExpChat.ai)
//  Historie:     2025-11-23  Erstfassung
//                2025-11-26  Typen/Fixes (Bias-Grad), sichere Updates
// ===========================================================================
use std::any::Any;

use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};

use crate::{adam::Adam, llm::Layer};

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct OutputProjection {
    #[bincode(with_serde)] pub w_out: Array2<f32>,   // [embedding_dim, vocab_size]
    #[bincode(with_serde)] pub b_out: Array2<f32>,   // [1, vocab_size]

    #[bincode(with_serde)]
    pub optimizer: Adam,

    #[serde(skip)]
    #[bincode(with_serde)]
    pub cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).expect("ungueltige Normalverteilung");

        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            b_out: Array2::<f32>::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
        }
    }

    fn parameter_count(&self) -> usize {
        let mut i_total: usize = 0;
        i_total += self.w_out.len();
        i_total += self.b_out.len();
        i_total
    }

    pub fn set_accumulate_steps(&mut self, steps: usize) {
        self.optimizer.set_accumulate_steps(steps);
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str { "OutputProjection" }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref()
            .expect("forward muss vor backward aufgerufen werden");

        let grad_w_out = input.t().dot(grads);
        let grad_b_out = grads.mean_axis(Axis(0)).expect("mean_axis schlug fehl"); // [vocab]
        let grad_b_out_2d = grad_b_out.insert_axis(Axis(0)); // [1, vocab]

        let grad_input = grads.dot(&self.w_out.t());

        self.optimizer.step(&mut self.w_out, &grad_w_out, lr);
        self.b_out -= &(lr * &grad_b_out_2d);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}
