// ===========================================================================
//  Datei:        layer_norm.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        Layer Normalisation
// ---------------------------------------------------------------------------
//  Autor:        Marcus Schlieper
//  Organisation: ExpChat.ai – Der KI-Chat-Client für den Mittelstand
//  Historie:
//      2025-11-23  Erstfassung
//      2025-11-26  Numerik-Fix (sqrt(var+eps)), saubere Backward-Formel,
//                  Caches für bincode/serde skip, Typspezifizierung f32
// ===========================================================================

use ndarray::{Array2, Axis};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;

use crate::{adam::Adam, llm::Layer};

/// Layer-Normalisierung mit lernbaren Parametern.
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct LayerNorm {
    epsilon: f32,

    // Lernbare Parameter
    #[bincode(with_serde)]
    gamma: Array2<f32>,
    #[bincode(with_serde)]
    beta:  Array2<f32>,

    // Caches: nur Laufzeit, nicht serialisieren
    #[serde(skip)]
    #[bincode(with_serde)]
    cached_input: Option<Array2<f32>>,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_mean:  Option<Array2<f32>>,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_std:   Option<Array2<f32>>,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_denom:   Option<Array2<f32>>,

    #[serde(skip)]
    #[bincode(with_serde)]
    cached_x_hat:   Option<Array2<f32>>,

    // Optimierer
    #[bincode(with_serde)]
    optimizer_gamma: Adam,
    #[bincode(with_serde)]
    optimizer_beta:  Adam,
}

// ---------------------------------------------------------------------------
//  Implementierung
// ---------------------------------------------------------------------------
impl LayerNorm {
    /// Erzeugt eine LayerNorm-Instanz mit γ = 1, β = 0.
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)),  // γ initial = 1
            beta:  Array2::zeros((1, embedding_dim)), // β initial = 0
            cached_input: None,
            cached_mean:  None,
            cached_std:  None,
            cached_denom: None,
            cached_x_hat: None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta:  Adam::new((1, embedding_dim)),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    // Batch-Akkumulation für beide Optimizer setzen (öffentlich)
    pub fn set_accumulate_steps(&mut self, steps: usize) {
        self.optimizer_gamma.set_accumulate_steps(steps);
        self.optimizer_beta.set_accumulate_steps(steps);
    }

    /// Vorwärts: Normalisierung + affine Transformation.
    /// Korrekt: denom = sqrt(var + epsilon)
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Mittelwert je Zeile (über Features)
        let mean = input
            .mean_axis(Axis(1))
            .expect("mean_axis schlug fehl")
            .insert_axis(Axis(1)); // [seq, 1]

        // Varianz je Zeile (über Features)
        let var = input
            .var_axis(Axis(1), 0.0) // ddof=0
            .insert_axis(Axis(1)); // [seq, 1]

        // denom = sqrt(var + eps)
        let denom = (var.mapv(|v| v + self.epsilon)).mapv(|v| v.sqrt()); // [seq, 1]

        // x_hat und Ausgabe
        let x_hat = (input - &mean) / &denom;          // [seq, embed]
        let out   = &self.gamma * &x_hat + &self.beta; // broadcast

        // Caches
        self.cached_input = Some(input.clone());
        self.cached_mean  = Some(mean);
        self.cached_denom = Some(denom);
        self.cached_x_hat = Some(x_hat);

        out
    }
}

// ---------------------------------------------------------------------------
//  Trait-Implementierung: Layer
// ---------------------------------------------------------------------------
impl Layer for LayerNorm {
    fn layer_type(&self) -> &str { "LayerNorm" }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    /// Stabiler Backward:
    /// dx = (1/denom) * (dyγ - mean(dyγ) - x_hat * mean(dyγ * x_hat))
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Caches holen
        let x      = self.cached_input.as_ref()
            .expect("forward muss vor backward aufgerufen werden");
        let mean   = self.cached_mean.as_ref().expect("mean nicht gesetzt");
        let denom  = self.cached_denom.as_ref().expect("denom nicht gesetzt");
        let x_hat  = self.cached_x_hat.as_ref().expect("x_hat nicht gesetzt");

        let seq_len = x.shape()[0];
        // let feat    = x.shape()[1] as f32;
        debug_assert_eq!(grads.shape(), &[seq_len, self.gamma.shape()[1]]);

        // dgamma, dbeta (Summe über Zeilen)
        let grad_gamma = (x_hat * grads).sum_axis(Axis(0)).insert_axis(Axis(0)); // [1, embed]
        let grad_beta  = grads.sum_axis(Axis(0)).insert_axis(Axis(0));           // [1, embed]

        // dyγ
        let dy_gamma = grads * &self.gamma; // [seq, embed]

        // Mittelwerte über Features (je Zeile -> [seq,1])
        let mean_dy_gamma = dy_gamma.mean_axis(Axis(1))
            .expect("mean_axis dyγ fehlgeschlagen")
            .insert_axis(Axis(1)); // [seq,1]

        let mean_dy_gamma_xhat = (dy_gamma.clone() * x_hat).mean_axis(Axis(1))
            .expect("mean_axis dyγ*x_hat fehlgeschlagen")
            .insert_axis(Axis(1)); // [seq,1]

        // dx gemäß Standard-Formel
        let dx = (dy_gamma
                 - &mean_dy_gamma
                 - x_hat * &mean_dy_gamma_xhat) / denom;

        // Parameter-Update via Adam
        self.optimizer_gamma.step(&mut self.gamma, &grad_gamma, lr);
        self.optimizer_beta .step(&mut self.beta , &grad_beta , lr);

        dx
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
}
