// ===========================================================================
//  Datei:        embeddings.rs
//  Projekt:      Lightweight Language Model (LLM)
//  Modul:        Embeddings
// ---------------------------------------------------------------------------
//  Beschreibung:
//      Dieses Modul stellt die Datenstrukturen sowie die Logik fuer Token-
//      und Positions-Embeddings bereit. Die Implementation basiert auf
//      N-dim Arrays (ndarray) und verwendet den Optimierer Adam zur
//      Parameteraktualisierung. Darueber hinaus erfuellt das Modul das
//      Layer-Trait, so dass es nahtlos in das Modellgraph-Design integriert
//      werden kann. Saemtliche Routinen sind darauf ausgelegt, speicher- und
//      laufzeiteffizient zu arbeiten, indem temporaere Objekte vermieden
//      werden. Fehlerbedingungen, etwa out-of-bounds Zugriffe, werden durch
//      explizite Plausibilitaetspruefungen abgefangen.
// ---------------------------------------------------------------------------
//  Autor:        Marcus Schlieper <mschlieper@ylook.de>
//  Organisation: ExpChat.ai, Breckerfeld
//  Telefon:      +49 2338 8748862 | Mobil: +49 151 1575 1864
//  Historie:
//      2025-11-23  Initialversion erstellt und dokumentiert.
//      2025-11-23  Kommentierung gem. Anforderung erweitert.
// ---------------------------------------------------------------------------
//  Sicherheitshinweis:
//      Alle panics dienen der fruehzeitigen Detektion logischer
//      Fehlkonfigurationen. In produktiven Umgebungen ist eine
//      feinere Fehlerbehandlung in Betracht zu ziehen.
// ===========================================================================

use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};
use bincode::{Encode, Decode};
use serde   ::{Serialize, Deserialize};
use std::any::Any;

// ---------------------------------------------------------------------------
//  Abhaengigkeiten innerhalb des Projektes
// ---------------------------------------------------------------------------
use crate::{
    EMBEDDING_DIM,
    MAX_SEQ_LEN,
    adam::Adam,
    llm::Layer,
};
use crate::tokenizer_bpe::Tokenizer;

// ---------------------------------------------------------------------------
//  Datenstruktur: Embeddings
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct Embeddings {
    /// Matrix der Token-Embeddings, Dimension: [vocab_size, embedding_dim]
    #[bincode(with_serde)]
    pub token_embeddings:      Array2<f32>,

    /// Matrix der Positions-Embeddings, Dimension: [max_seq_len, embedding_dim]
    #[bincode(with_serde)]
    pub positional_embeddings: Array2<f32>,

    /// Zwischenspeicher fuer das Eingabetensor im Forward-Pass
    #[serde(skip)]
    #[bincode(with_serde)]
    pub cached_input: Option<Array2<f32>>,   // nur Laufzeit, nicht serialisieren

    /// Adam-Optimierer fuer Token-Embeddings
    #[bincode(with_serde)]
    pub token_optimizer:      Adam,

    /// Adam-Optimierer fuer Positions-Embeddings
    #[bincode(with_serde)]
    pub positional_optimizer: Adam,
}

// ---------------------------------------------------------------------------
//  Default-Implementierung
// ---------------------------------------------------------------------------
impl Default for Embeddings {
    /// Erzeugt ein Embeddings-Objekt mit Standard-Vokabular
    fn default() -> Self {
        // Vokabular inkl. UNK und EOS
        let vocab_size = Tokenizer::new_byte_level().vocab_size();


        Self {
            token_embeddings:      Self::init_embeddings(vocab_size, EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input:          None,
            token_optimizer:       Adam::new((vocab_size, EMBEDDING_DIM)),
            positional_optimizer:  Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }
}

// ---------------------------------------------------------------------------
//  Oeffentliche Methoden
// ---------------------------------------------------------------------------
impl Embeddings {

    /// Erzeugt ein Embedding-Objekt auf Basis eines beliebigen Vokabulars.
     pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let vocab_size = tokenizer.vocab_size();
        Self {
            token_embeddings: Self::init_embeddings(vocab_size, EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((vocab_size, EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }

    // -----------------------------------------------------------------------
    //  Initialisierer
    // -----------------------------------------------------------------------

    /// Initialisiert Token-Embeddings mit Werten aus einer Normalverteilung.
    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng     = rand::thread_rng();
        let normal_dist = Normal::new(0.0, 0.02).expect("Ungueltige Normalverteilung");
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal_dist.sample(&mut rng))
    }

    /// Initialisiert Positions-Embeddings analog zu `init_embeddings`.
    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng     = rand::thread_rng();
        let normal_dist = Normal::new(0.0, 0.02).expect("Ungueltige Normalverteilung");
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| normal_dist.sample(&mut rng))
    }

    // -----------------------------------------------------------------------
    //  Hilfsfunktionen (private)
    // -----------------------------------------------------------------------

    /// Liefert Embeddings fuer eine Liste von Token-IDs.
    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    embeddings.nrows()
                );
            }
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    /// Schneidet die Positions-Embedding-Matrix auf die Sequenzlaenge zu.
    fn get_positional_embeddings(
        positional_encodings: &Array2<f32>,
        seq_len: usize,
    ) -> Array2<f32> {
        if seq_len > positional_encodings.nrows() {
            panic!(
                "Sequence length {} exceeds maximum {}",
                seq_len,
                positional_encodings.nrows()
            );
        }
        positional_encodings.slice(s![0..seq_len, ..]).to_owned()
    }

    // -----------------------------------------------------------------------
    //  Oeffentliche Fachfunktion
    // -----------------------------------------------------------------------

    /// Additive Kombination aus Token- und Positions-Embeddings.
    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        let position_embeds =
            Self::get_positional_embeddings(&self.positional_embeddings, token_ids.len());
        token_embeds + position_embeds // Elementweises Addieren
    }
}

// ---------------------------------------------------------------------------
//  Trait-Implementierung: Layer
// ---------------------------------------------------------------------------
impl Layer for Embeddings {
    /// Rueckmeldung des Typs fuer Logging / Serialisierung
    fn layer_type(&self) -> &str { "Embeddings" }

    fn as_any(&self) -> &dyn Any       { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    /// Forward-Pass: wandelt Token-IDs in Embeddings um.
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Erwartete Eingabeform: [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&token_ids) // Rueckgabeform: [sequence_length, embedding_dim]
    }

    /// Backward-Pass: berechnet Gradienten und aktualisiert Parameter.
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input     = self.cached_input.as_ref().expect("Forward muss vor Backward erfolgen");
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads     = grads.view(); // (sequence_length, embedding_dim)

        // Gradientenspeicher initialisieren
        let mut token_grads      = Array2::<f32>::zeros(self.token_embeddings.dim());
        let mut positional_grads = Array2::<f32>::zeros(self.positional_embeddings.dim());

        // Gradienten ueber die Sequenz akkumulieren
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.token_embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    self.token_embeddings.nrows()
                );
            }
            let grad_row = grads.row(i);

            // Token-Gradient
            {
                let mut token_row = token_grads.row_mut(token_id);
                token_row += &grad_row;
            }

            // Positions-Gradient
            {
                let mut pos_row = positional_grads.row_mut(i);
                pos_row += &grad_row;
            }
        }

        // Parameteraktualisierung via Adam
        self.token_optimizer
            .step(&mut self.token_embeddings,      &token_grads,      lr);
        self.positional_optimizer
            .step(&mut self.positional_embeddings, &positional_grads, lr);

        // Rueckgabe des Gradienten fuer vorgelagerte Ebenen
        grads.to_owned()
    }

    /// Anzahl lernbarer Parameter (fuer Monitoring / Logging)
    fn parameters(&self) -> usize {
        self.token_embeddings.len() + self.positional_embeddings.len()
    }
}
