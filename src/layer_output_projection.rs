// ===========================================================================
//  Datei:        layer_output_projections.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        Output Projection Layer
// ---------------------------------------------------------------------------
//  Beschreibung:
//      Dieses Modul stellt die finale Projektionsschicht eines
//      Sprachmodells bereit.  Die Schicht transformiert die Ausgaenge der
//      letzten Embedding- bzw. Decoder­ebene (Dimension embedding_dim) in
//      unskalierte Logits der Groesse vocab_size.  Die Gewichte werden mit
//      Xavier/He-Initialisierung erzeugt, die Optimierung erfolgt mittels
//      Adam.  Saemtliche Zwischenergebnisse des Forward-Passes werden
//      zwischengespeichert, um im Backward-Pass Gradienten effizient
//      berechnen zu koennen.
//
//  Autor:        Marcus Schlieper
//  Organisation: ExpChat.ai, Breckerfeld
//  Telefon:      +49 2338 8748862  |  Mobil: +49 151 1575 1864
//  Historie:
//      2025-11-23  Erstfassung erstellt und dokumentiert.
// ---------------------------------------------------------------------------
//  Sicherheitshinweis:
//      Ein Panic tritt auf, wenn vor dem Aufruf von `backward` kein
//      `forward` ausgefuehrt wurde (cached_input == None).  Dies stellt eine
//      fehlerhafte Aufrufreihenfolge dar und muss waehrend der Entwicklung
//      abgefangen werden.
// ===========================================================================

use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;

use crate::{adam::Adam, llm::Layer};

/// Output-Projektionsschicht:  Embeddings -> Logits
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct OutputProjection {
    // ---------------------------------------------------------------
    // Lernbare Parameter
    // ---------------------------------------------------------------
   #[bincode(with_serde)] pub w_out: Array2<f32>,   // Gewichtsmatrix  [embedding_dim, vocab_size]

    #[bincode(with_serde)] pub b_out: Array2<f32>,   // Bias-Vektor     [1, vocab_size]

    // ---------------------------------------------------------------
    // Optimierer
    // ---------------------------------------------------------------
    #[bincode(with_serde)]
    pub optimizer: Adam, // Adam-Instanz fuer w_out (Bias wird direkt angepasst)

    // ---------------------------------------------------------------
    // Zwischenspeicher (nur Laufzeit, nicht serialisieren)
    // ---------------------------------------------------------------
   #[serde(skip)]#[bincode(with_serde)]  pub cached_input: Option<Array2<f32>>, // Eingabe des Forward-Passes
}

// ---------------------------------------------------------------------------
//  Oeffentliche Schnittstelle
// ---------------------------------------------------------------------------
impl OutputProjection {
    /// Erstellt eine neue Ausgabeschicht.
    ///
    /// * `embedding_dim` – Dimensionalitaet der Eingangsembeddings  
    /// * `vocab_size`    – Groesse des Zielvokabulars
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        // Zufallszahlgenerator initialisieren
        let mut rng = rand::thread_rng();

        // Xavier/He-Initialisierung: std = sqrt(2 / fan_in)
        let std     = (2.0 / embedding_dim as f32).sqrt();
        let normal  = Normal::new(0.0, std)
            .expect("ungueltige Normalverteilung fuer OutputProjection");

        OutputProjection {
            w_out: Array2::from_shape_fn(
                (embedding_dim, vocab_size),
                |_| normal.sample(&mut rng),
            ),
            b_out: Array2::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
        }
    }
}

// ---------------------------------------------------------------------------
//  Trait-Implementierung: Layer
// ---------------------------------------------------------------------------
impl Layer for OutputProjection {
    /// Rueckmeldung des Schichttyps
    fn layer_type(&self) -> &str { "OutputProjection" }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    /// Forward-Pass:  Embeddings -> Logits
    ///
    /// Eingabeform:  [sequence_len, embedding_dim]  
    /// Rueckgabe   : [sequence_len, vocab_size]
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Eingabe fuer Backward zwischenspeichern
        self.cached_input = Some(input.clone());

        // Matrixmultiplikation + Bias
        input.dot(&self.w_out) + &self.b_out
    }

    /// Backward-Pass: berechnet Gradienten und aktualisiert Parameter
    ///
    /// * `grads` – eingehender Gradient [sequence_len, vocab_size]  
    /// * `lr`    – Lernrate
    ///
    /// Rueckgabe: Gradient fuer vorgelagerte Schicht
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // ------------ Zwischenspeicher abrufen --------------------
        let input = self.cached_input
            .as_ref()
            .expect("forward muss vor backward aufgerufen werden");

        // ------------ Gradienten berechnen ------------------------
        let grad_w_out = input.t().dot(grads);               // dL/dW
        let grad_b_out = grads.mean_axis(Axis(0))            // dL/db
                               .expect("mean_axis schlug fehl");

        let grad_input = grads.dot(&self.w_out.t());         // Gradient fuer vorherige Ebene

        // ------------ Parameter-Update ---------------------------
        self.optimizer.step(&mut self.w_out, &grad_w_out, lr);
        self.b_out -= &(lr * &grad_b_out);                   // Bias via SGD

        grad_input
    }

    /// Gesamtanzahl lernbarer Parameter
    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}
