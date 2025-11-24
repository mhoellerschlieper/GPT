// ===========================================================================
//  Datei:        feed_forward.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        FeedForward Layer
// ---------------------------------------------------------------------------
//  Beschreibung:
//      Dieses Modul implementiert eine zweistufige, residual gekoppelte
//      Feed-Forward-Schicht, bestehend aus
//
//          1) Linear  :  input · w1 + b1
//          2) Aktivierung: ReLU
//          3) Linear  :  hidden · w2 + b2
//          4) Residual:  output + input
//
//      Die Gewichte werden mittels Xavier/He-Initialisierung erzeugt,
//      die Parameteroptimierung erfolgt über den Adam-Optimierer.
//      Alle internen Zwischenergebnisse des Forward-Passes werden
//      zwischengespeichert, sodass der Backward-Pass die benötigten
//      Gradienten ohne Rekonstruktion berechnen kann.
// ---------------------------------------------------------------------------
//  Autor:        Marcus Schlieper
//  Organisation: ExpChat.ai, Breckerfeld
//  Telefon:      +49 2338 8748862 | Mobil: +49 151 1575 1864
//  E-Mail:       mschlieper@ylook.de
//  Zusatzinfo:   Der KI Chat Client fuer den Mittelstand aus Breckerfeld.
//  Historie:
//      2025-11-23  Initialversion erstellt und vollstaendig dokumentiert.
// ---------------------------------------------------------------------------
//  Sicherheitshinweis:
//      Panics werden nur zur Erkennung logischer Fehlkonfigurationen
//      eingesetzt. In produktiven Umgebungen sollte eine feinere
//      Fehlerbehandlung erfolgen.
// ===========================================================================

use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;

use crate::{adam::Adam, llm::Layer};

/// Feed-Forward-Schicht mit zwei Linear-ReLU-Linear Blöcken
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct FeedForward {
    // -------------------------------------------------------------------
    //  Lernbare Parameter
    // -------------------------------------------------------------------
    #[bincode(with_serde)] pub w1: Array2<f32>, // Gewichtsmatrix 1  [embedding_dim, hidden_dim]
    #[bincode(with_serde)] pub b1: Array2<f32>, // Bias‐Vektor 1     [1, hidden_dim]
    #[bincode(with_serde)] pub w2: Array2<f32>, // Gewichtsmatrix 2  [hidden_dim, embedding_dim]
    #[bincode(with_serde)] pub b2: Array2<f32>, // Bias‐Vektor 2     [1, embedding_dim]

    // -------------------------------------------------------------------
    //  Zwischenspeicher (nur Laufzeit, nicht serialisieren)
    // -------------------------------------------------------------------
    #[serde(skip)] #[bincode(with_serde)] pub input:                  Option<Array2<f32>>,
    #[serde(skip)] #[bincode(with_serde)] pub hidden_pre_activation:  Option<Array2<f32>>,
    #[serde(skip)] #[bincode(with_serde)] pub hidden_post_activation: Option<Array2<f32>>,

    // -------------------------------------------------------------------
    //  Optimierer
    // -------------------------------------------------------------------
    #[bincode(with_serde)] pub optimizer_w1: Adam,
    #[bincode(with_serde)] pub optimizer_b1: Adam,
    #[bincode(with_serde)] pub optimizer_w2: Adam,
    #[bincode(with_serde)] pub optimizer_b2: Adam,
}

impl FeedForward {
    /// Erstellt eine Feed-Forward-Schicht mit zufällig initialisierten Gewichten
    ///
    /// * `embedding_dim` – Eingabedimension sowie Ausgabedimension (residual)
    /// * `hidden_dim`    – Anzahl versteckter Neuronen zwischen den Linearschichten
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Zufallszahlgenerator (thread-sicher)
        let mut rng = rand::thread_rng();

        // Xavier/He-Initialisierung für w1
        let std_w1    = (2.0 / embedding_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1)
            .expect("ungueltige Normalverteilung fuer w1");

        // Xavier/He-Initialisierung für w2
        let std_w2    = (2.0 / hidden_dim as f32).sqrt();
        let normal_w2 = Normal::new(0.0, std_w2)
            .expect("ungueltige Normalverteilung fuer w2");

        FeedForward {
            // ----------------------------- Parameter --------------------
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim),
                                      |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, hidden_dim)),
            w2: Array2::from_shape_fn((hidden_dim, embedding_dim),
                                      |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, embedding_dim)),

            // -------------------------- Zwischenspeicher ----------------
            input:                  None,
            hidden_pre_activation:  None,
            hidden_post_activation: None,

            // ----------------------------- Optimierer -------------------
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_b1: Adam::new((1,           hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim,  embedding_dim)),
            optimizer_b2: Adam::new((1,           embedding_dim)),
        }
    }
}

// ---------------------------------------------------------------------------
//  Trait-Implementierung: Layer
// ---------------------------------------------------------------------------
impl Layer for FeedForward {
    /// Liefert den Ebenentyp als Zeichenkette
    fn layer_type(&self) -> &str { "FeedForward" }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    /// Backward-Pass: berechnet Gradienten und aktualisiert Parameter
    ///
    /// * `grads` – eingehender Gradient aus nachgelagerter Schicht
    /// * `lr`    – Lernrate für den Adam-Schritt
    ///
    /// Rückgabewert: Gradient für vorgelagerte Schicht (inkl. Residual)
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // ----------------------------- Caching ------------------------
        let input                 = self.input
                                         .as_ref()
                                         .expect("forward muss zuerst ausgefuehrt werden");
        let hidden_pre_activation = self.hidden_pre_activation
                                         .as_ref()
                                         .expect("kein hidden_pre_activation vorhanden");
        let hidden_post_activation = self.hidden_post_activation
                                         .as_ref()
                                         .expect("kein hidden_post_activation vorhanden");

        // --------------------------- Gradienten -----------------------
        // 1) dL/dW2  und  dL/db2
        let grad_w2 = hidden_post_activation.t().dot(grads);
        let grad_b2 = grads.sum_axis(Axis(0))
                           .insert_axis(Axis(0)); // Form: [1, embedding_dim]

        // 2) dL/dHidden_post
        let grad_hidden_post = grads.dot(&self.w2.t());

        // 3) Rueckpropagation durch ReLU
        let relu_grad = hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad_hidden_pre = grad_hidden_post * relu_grad;

        // 4) dL/dW1  und  dL/db1
        let grad_w1 = input.t().dot(&grad_hidden_pre);
        let grad_b1 = grad_hidden_pre.sum_axis(Axis(0))
                                     .insert_axis(Axis(0)); // Form: [1, hidden_dim]

        // 5) Gradient bezüglich Eingang der Schicht
        let grad_input_feed = grad_hidden_pre.dot(&self.w1.t());

        // 6) Residual-Verbindung addieren
        let grad_input = grad_input_feed + grads;

        // ----------------------- Parameter-Update ---------------------
        self.optimizer_w2.step(&mut self.w2, &grad_w2, lr);
        self.optimizer_b2.step(&mut self.b2, &grad_b2, lr);
        self.optimizer_w1.step(&mut self.w1, &grad_w1, lr);
        self.optimizer_b1.step(&mut self.b1, &grad_b1, lr);

        grad_input
    }

    /// Forward-Pass: berechnet Schichtausgabe und speichert Zwischenergebnisse
    ///
    /// * `input` – Eingangstensor der Form [sequence_len, embedding_dim]
    ///
    /// Rückgabewert: Ausgangstensor gleicher Form
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 1) Linear + Bias
        let hidden_pre = input.dot(&self.w1) + &self.b1;

        // 2) ReLU-Aktivierung
        let hidden_post = hidden_pre.mapv(|x| x.max(0.0));

        // 3) Zweite Linearschicht
        let output_ff = hidden_post.dot(&self.w2) + &self.b2;

        // --------------------------- Caching --------------------------
        self.input                  = Some(input.clone());
        self.hidden_pre_activation  = Some(hidden_pre);
        self.hidden_post_activation = Some(hidden_post);

        // 4) Residual-Verbindung
        output_ff + input
    }

    /// Anzahl aller lernbaren Parameter (Gewichte + Bias)
    fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}
