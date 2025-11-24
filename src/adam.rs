// =============================================
// adam.rs
// =============================================

#![forbid(unsafe_code)]               // Gewährleistet ausschließliche Nutzung von Safe Rust

// --- Externe Abhängigkeiten -----------------------------------------------
use ndarray::Array2;                  // Dicht besetzte 2-D-Arrays
use bincode::{Encode, Decode};        // Binäre Serialisierung
use serde   ::{Serialize, Deserialize};// Mensch- und maschinenlesbare Serialisierung

// --- Datentyp --------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Encode, Decode)]
/// Struktur, die den internen Zustand des Adam-Optimierers kapselt.
///
/// Feldbedeutungen  
/// • `beta1`, `beta2` – Exponentielle Abklingraten für erste und zweite Momentabschätzung  
/// • `epsilon`        – Numerischer Stabilisator zur Vermeidung einer Division durch 0  
/// • `timestep`       – Anzahl bislang ausgeführter Optimierungsschritte (Start bei 0)  
/// • `m`              – Schätzer des ersten Moments (Mittelwert der Gradienten)  
/// • `v`              – Schätzer des zweiten Moments (ungefähr Varianz der Gradienten)
pub struct Adam {
    beta1:    f32,
    beta2:    f32,
    epsilon:  f32,
    timestep: usize,

    #[bincode(with_serde)]            // Serialisierungsstrategie für ndarray
    pub m: Array2<f32>,
    #[bincode(with_serde)]
    pub v: Array2<f32>,
}

// --- assoziierte Funktionen -----------------------------------------------
impl Adam {

    /// Erzeugt eine neue Instanz mit Nullinitialisierung der Momentabschätzungen.
    ///
    /// Parameter  
    /// • `shape` – Tupel `(rows, cols)`, das die Dimensionen der zu optimierenden
    ///   Parametermatrix festlegt.  
    ///
    /// Rückgabe  
    /// • Instanz von `Adam` mit Standard-Hyperparametern
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,               // Standard gemäß Originalpublikation
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),  // Erste Momentabschätzung (m₀)
            v: Array2::zeros(shape),  // Zweite Momentabschätzung (v₀)
        }
    }

    /// Führt einen einzelnen Optimierungsschritt durch.
    ///
    /// Parameter  
    /// • `params` – Mutabler Verweis auf die Parametermatrix θ ∈ ℝ^{m×n}  
    /// • `grads`  – Gradientenmatrix ∇L(θ) gleicher Dimension  
    /// • `lr`     – Lernrate α (> 0)  
    ///
    /// Effekte  
    /// • Aktualisiert interne Momentabschätzungen `m` und `v`  
    /// • Erhöht `timestep` (Zeitschritt t) um 1  
    /// • Modifiziert `params` in-place gemäß Adam-Update-Regel  
    ///
    /// Stabilität  
    /// • Alle Operationen bleiben im Bereich der Gleitkommazahlen `f32`  
    /// • `epsilon` verhindert Division durch 0
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        // Zeitschritt erhöhen
        self.timestep += 1;

        // 1. Momentum m_t = β₁·m_{t-1} + (1 − β₁)·g_t
        self.m = &self.m * self.beta1 + &(grads * (1.0 - self.beta1));

        // 2. Varianz v_t = β₂·v_{t-1} + (1 − β₂)·(g_t ⊙ g_t)
        self.v = &self.v * self.beta2 + &(grads.mapv(|x| x * x) * (1.0 - self.beta2));

        // 3. Bias-Korrektur
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.timestep as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.timestep as i32));

        // 4. Parameteraktualisierung  θ ← θ − α·m̂_t /(√v̂_t + ε)
        //    (Clone überflüssig, da nur geliehene Werte)
        let update = m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon);
        *params -= &(update * lr);    // In-place-Subtraktion
    }
}
