# Temporal Transformer Rust — Forschungscodebasis

## Überblick
Dieses Repository stellt eine referenz- und forschungsorientierte Implementierung eines vollkommen in **Rust** verfassten Transformermodells bereit, das eine zeitsensitive Erweiterung traditioneller Architekturen um **kontinuierliche Zeit-Embeddings**, **FlashAttention-ähnliche Blockaggregation**, **Mixed-Precision-Berechnung** sowie **Gradient Checkpointing** beinhaltet.  
Die Codebasis adressiert Expert:innen aus den Bereichen Maschinelles Lernen, Systemprogrammierung und High-Performance-Computing, welche an reproduzierbaren Experimenten zur Effizienzsteigerung grosser Sprach- und Sequenzmodelle interessiert sind.

---

## Funktionsumfang
| Kategorie                              | Merkmal                                                                                                       |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Zeitmodellierung                       | `Time2Vec`-Einbettung (Kazemi & Mehri, 2019) plus Δt-Bias in der Self-Attention                                |
| Selbstaufmerksamkeit                   | Multi-Head Attention mit Rotary-Positional-Encoding und Key/Value-Ringpuffer für O(1)-Dekodierung             |
| Speicher- und Rechenoptimierung        | FlashAttention-ähnliche Blockverarbeitung (Dao et al., 2022) sowie Gradient Checkpointing                     |
| Numerische Präzision                   | Umschaltbares `FloatDType`-Trait (f32/bf16/f16)                                                               |
| Regulierung                            | Konfigurierbares Drop-out auf Kopf-, Feed-Forward- und Residualebene                                          |
| Modularität                            | Strikte Trennung in Rust-Module (`multi_head_attention`, `feed_forward_geglu`, `time2vec`, usw.)              |

---

## Installationshinweise
```bash
git clone https://github.com/mhoellerschlieper/GPT.git
cd temporal-transformer-rust
cargo build --release  # setzt Rust 1.75+ sowie BLAS-unterstützendes ndarray voraus
