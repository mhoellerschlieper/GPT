// ===========================================================================
//  Datei:        layer_norm.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        Layer Normalisation
// ---------------------------------------------------------------------------
//  Beschreibung:
//      Dieses Modul stellt eine vollständig differenzierbare Implementierung
//      der Layer-Normalisierung (LayerNorm) inklusive lernbarer Skalierungs-
//      (γ) und Verschiebungsparameter (β) bereit. Die Schicht normalisiert
//      jede Eingabezeile (Token-Vektor) auf Mittelwert = 0 und Varianz = 1,
//      bevor eine affine Transformation erfolgt.  Die Realisierung
//      unterstützt Vorwärts- und Rückwärtsdurchlauf und fügt sich über das
//      Trait `Layer` nahtlos in den gesamten Modellgraphen ein.  Die
//      Parameteraktualisierung wird durch einen Adam-Optimierer vorgenommen.
// ---------------------------------------------------------------------------
//  Autor:        Marcus Schlieper
//  Organisation: ExpChat.ai – Der KI-Chat-Client für den Mittelstand aus
//                Breckerfeld im Sauerland
//  Kontakt:      Tel  : +49 2338 8748862
//                Mobil: +49 151 1575 1864
//                Mail : mschlieper@ylook.de
//  Adresse:      Epscheider Str. 21, 58339 Breckerfeld, Deutschland
//  Historie:
//      2025-11-23  Erstfassung erstellt und umfassend dokumentiert.
// ---------------------------------------------------------------------------
//  Sicherheitshinweis:
//      Laufzeitfehler (z. B. Division durch Null) werden durch einen
//      stabilen Epsilon-Wert verhindert.  Panics treten ausschließlich bei
//      inkorrekter interner Nutzung (fehlender Forward-Aufruf) auf und
//      deuten auf einen Programmierfehler hin.
// ===========================================================================

use ndarray::{Array2, Axis};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;

use crate::{adam::Adam, llm::Layer};

/// Layer-Normalisierung mit lernbaren Parametern.
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct LayerNorm {
    /// Numerischer Stabilisator zur Vermeidung von Division-durch-Null.
    epsilon: f32,

    // ----------------------------- Parameter ------------------------------
    /// Skalierungsfaktor γ Form: [1, embedding_dim]
   #[bincode(with_serde)] gamma: Array2<f32>,
    /// Verschiebung β Form: [1, embedding_dim]
   #[bincode(with_serde)] beta:  Array2<f32>,


    // --------------------------- Zwischenspeicher -------------------------
    /// Eingangstensor des Forward-Passes (nur Laufzeit)
    #[serde(skip)]#[bincode(with_serde)]  cached_input: Option<Array2<f32>>,
    /// Mittelwert pro Tokenzeile (nur Laufzeit)
  #[serde(skip)]#[bincode(with_serde)]  cached_mean:  Option<Array2<f32>>,
    /// Standardabweichung pro Tokenzeile (nur Laufzeit)
  #[serde(skip)]#[bincode(with_serde)]  cached_std:   Option<Array2<f32>>,

    // ----------------------------- Optimierer -----------------------------
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
    ///
    /// * `embedding_dim` – Dimension der einzeln zu normalisierenden Vektoren.
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)),  // γ initial = 1
            beta:  Array2::zeros((1, embedding_dim)), // β initial = 0
            cached_input: None,
            cached_mean:  None,
            cached_std:   None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta:  Adam::new((1, embedding_dim)),
        }
    }

    /// Führt die eigentliche Normalisierung und anschließende affine
    /// Transformation aus; Zwischenergebnisse werden für den Backward-Pass
    /// zwischengespeichert.
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Mittelwert und Standardabweichung über Achse 1 (Features je Token).
        let mean = input.mean_axis(Axis(1))
                        .expect("mean_axis schlug fehl")
                        .insert_axis(Axis(1));
        let std  = input.std_axis(Axis(1), 0.0)  // 0.0 = ddof
                        .insert_axis(Axis(1));

        // Zwischenspeicher für Rückwärtsrechnung
        self.cached_input = Some(input.clone());
        self.cached_mean  = Some(mean.clone());
        self.cached_std   = Some(std.clone());

        // Normalisierung + affine Transformation
        let normalized = (input - &mean) / (&std + self.epsilon);
        &self.gamma * &normalized + &self.beta
    }
}

// ---------------------------------------------------------------------------
//  Trait-Implementierung: Layer
// ---------------------------------------------------------------------------
impl Layer for LayerNorm {
    fn layer_type(&self) -> &str { "LayerNorm" }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    // ----------------------------- Forward --------------------------------
    /// Delegiert an `normalize`.
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    // ----------------------------- Backward -------------------------------
    /// Führt vollständige Rückwärtsableitung inkl. Parameter-Updates durch.
    ///
    /// * `grads` – Eingehender Gradienten-Tensor gleicher Form wie `input`.
    /// * `lr`    – Lernrate für den Adam-Optimierer.
    ///
    /// Rückgabewert: Gradienten-Tensor für vorgelagerte Schichten.
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // ------------------ Zwischengrößen abrufen ------------------------
        let input = self.cached_input
                        .as_ref()
                        .expect("forward muss vor backward aufgerufen werden");
        let mean  = self.cached_mean
                        .as_ref()
                        .expect("mean nicht gesetzt");
        let std   = self.cached_std
                        .as_ref()
                        .expect("std nicht gesetzt");

        // Vorab berechnete Normalisierung erneut herstellen
        let normalized   = (input - mean) / (std + self.epsilon);
        let n_features   = input.shape()[1] as f32;

        // 1) Gradienten bezüglich γ und β
        let grad_gamma   = (&normalized * grads)
                            .sum_axis(Axis(0))
                            .insert_axis(Axis(0));
        let grad_beta    = grads.sum_axis(Axis(0))
                            .insert_axis(Axis(0));

        // 2) Gradient w.r.t. normalisierte Werte
        let grad_norm    = &self.gamma * grads;

        // 3) Vollständige Ableitung nach Eingang (vgl. Ba et al., 2016)
        let variance     = std * std + self.epsilon;
        let grad_var     = (&grad_norm * &normalized)
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1))
                            * (-0.5)
                            / variance.mapv(|x| x.sqrt() * x.sqrt());
        let grad_mean    = grad_norm.sum_axis(Axis(1))
                            .insert_axis(Axis(1)) * (-1.0)
                            / (std + self.epsilon)
                          + &grad_var * (input - mean)
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1)) * (-2.0) / n_features;

        let grad_input   =  grad_norm / (std + self.epsilon)
                          + &grad_var * 2.0 * (input - mean) / n_features
                          + &grad_mean / n_features;

        // 4) Parameter-Update via Adam
        self.optimizer_gamma.step(&mut self.gamma, &grad_gamma, lr);
        self.optimizer_beta .step(&mut self.beta , &grad_beta , lr);

        grad_input
    }

    /// Gesamtanzahl lernbarer Parameter (γ + β).
    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
}
