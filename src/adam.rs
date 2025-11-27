// =============================================
// adam.rs
// =============================================

#![forbid(unsafe_code)]               // Gewährleistet ausschließliche Nutzung von Safe Rust

// --- Externe Abhängigkeiten -----------------------------------------------
use ndarray::{Array2, Zip};           // Dicht besetzte 2-D-Arrays
use bincode::{Encode, Decode};        // Binäre Serialisierung
use serde   ::{Serialize, Deserialize};// Mensch- und maschinenlesbare Serialisierung

// --- Datentyp --------------------------------------------------------------
/// Struktur, die den internen Zustand des Adam-Optimierers kapselt.
///
/// Feldbedeutungen  
/// • `beta1`, `beta2` – Exponentielle Abklingraten für erste und zweite Momentabschätzung  
/// • `epsilon`        – Numerischer Stabilisator zur Vermeidung einer Division durch 0  
/// • `timestep`       – Anzahl bislang ausgeführter Optimierungsschritte (Start bei 0)  
/// • `m`              – Schätzer des ersten Moments (Mittelwert der Gradienten)  
/// • `v`              – Schätzer des zweiten Moments (ungefähr Varianz der Gradienten)
#[derive(serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct Adam {
    #[bincode(with_serde)]
    m: Array2<f32>,
    #[bincode(with_serde)]
    v: Array2<f32>,
    t: usize,
    // Neu: Akkumulation
    i_accumulate: usize,             // Batch-Größe (>=1)
    i_since_update: usize,           // Schritte seit letztem Update
    #[bincode(with_serde)]
    grad_buf: Array2<f32>,           // Puffer für aufsummierte Gradienten
    // Hyperparameter
    beta1: f32,
    beta2: f32,
    eps: f32,
}

// --- assoziierte Funktionen -----------------------------------------------
impl Adam {
    pub fn new(shape: (usize, usize)) -> Self {
        Adam {
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            t: 0,
            i_accumulate: 1,
            i_since_update: 0,
            grad_buf: Array2::zeros(shape),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    // Neu: setze Batch-Größe für Akkumulation
    pub fn set_accumulate_steps(&mut self, steps: usize) {
        self.i_accumulate = steps.max(1);
        self.i_since_update = 0;
        self.grad_buf.fill(0.0);
    }

    // Schritt mit Akkumulation
    // Update erfolgt nur, wenn i_since_update + 1 == i_accumulate.
    pub fn step(&mut self, w: &mut Array2<f32>, grad: &Array2<f32>, lr: f32) {
        // Gradienten sammeln
        Zip::from(&mut self.grad_buf)
            .and(grad)
            .for_each(|gb, &g| *gb += g);

        self.i_since_update += 1;

        if self.i_since_update < self.i_accumulate {
            // Noch kein Update, erst sammeln
            return;
        }

        // Jetzt Update: gemittelter Grad
        let scale = 1.0 / (self.i_accumulate as f32);
        let mut g_avg = self.grad_buf.clone();
        g_avg.mapv_inplace(|x| x * scale);

        // Standard-Adam
        self.t += 1;
        let t = self.t as f32;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;

        Zip::from(&mut self.m)
            .and(&g_avg)
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g);
        Zip::from(&mut self.v)
            .and(&g_avg)
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * g * g);

        let bias_c1 = 1.0 - b1.powf(t);
        let bias_c2 = 1.0 - b2.powf(t);

        // w = w - lr * m_hat / (sqrt(v_hat) + eps)
        for ((w_ij, m_ij), v_ij) in w.iter_mut().zip(self.m.iter()).zip(self.v.iter()) {
            let m_hat = *m_ij / bias_c1;
            let v_hat = *v_ij / bias_c2;
            *w_ij -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Puffer leeren für nächste Akkumulation
        self.grad_buf.fill(0.0);
        self.i_since_update = 0;
    }
}