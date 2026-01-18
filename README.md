# ExpChat.ai LLM (Rust) - KI Chat Client fuer den Mittelstand

## Uebersicht

Dieses Repository enthaelt eine in Rust implementierte, experimentelle LLM Pipeline mit eigenem Byte-level BPE Tokenizer, einem Transformer Modell und einer CLI gesteuerten Trainings- und Inferenzumgebung. Der Fokus liegt auf einer pragmatischen, nachvollziehbaren End-to-End Kette von der Korpusaufbereitung bis zur interaktiven Nutzung, wobei robuste Defaults, defensives Fehlerhandling sowie reproduzierbare Artefakte (Tokenizer Persistenz, Checkpoints, Datensatzpakete) im Vordergrund stehen.

Das Projekt adressiert zentrale Anwendungsbereiche im Mittelstand, insbesondere Chat-basierte Assistenz, Wissensmanagement, sowie als technische Grundlage fuer RPA und KI Agents. Es handelt sich um ein forschungsnahes System, das bewusst transparent gehalten ist und dadurch eine gute Basis fuer Erweiterungen (z.B. Retrieval-Augmented Generation, Evaluationsframeworks, Data Governance) bietet.

## Funktionsumfang (High Level)

- Transformer LLM in Rust (f32, ndarray), mit modularer Layer-Architektur.
- Byte-level BPE Tokenizer mit reversibler Byte-Kodierung (verlustfreie Roundtrips Text <-> Tokens).
- CLI mit Menu fuer:
  - Training (zweiphasig: Pretraining plus Main Training)
  - Inferenz (interaktiv) und Inference Grid
  - Checkpoint Save/Load
  - Corpus Pipeline: fetch, prepare, filter, pack
- Korpus Pipeline mit konservativen Qualitaetsfiltern, globaler Deduplication und Leakage-reduzierendem Split nach Quellen (Source-IDs).
- Trainingsfeatures:
  - stabile Softmax (log-sum-exp)
  - AdamW Optimizer
  - Gradient Clipping
  - optionale GradScaler Logik
  - optionales Loss Masking fuer Chat-Formate (Assistant-only)

## Projektstruktur

- `src/main.rs`
  - CLI Entry Point, Menu, Trainings- und Corpus-Ablauf.
- `src/train.rs`
  - Training, Validierung, Inferenz, Sampling, Stop-Sequenzen, Sliding-Window Logik.
- `src/layers.rs`
  - Model Layers (Embeddings, TransformerBlockV2, Attention, FFN, Norm, Output Projection).
- `src/tokenize.rs`
  - Byte-level BPE Tokenizer (reversibel), Merge Training, Persistenz.
- `src/math.rs`
  - Numerische Utilities (AdamW, stabile Softmax, Gradient Clipping, GradScaler).
- `src/corpus.rs`
  - Corpus Pipeline (fetch/prepare/filter/pack) und Datenpaket-Erzeugung.
- `src/utils.rs`
  - Konfigurationen und Dataset Loader.

## Schnellstart

### Voraussetzungen

- Rust Toolchain (stable)
- Internetzugang (optional, fuer Corpus Fetch)
- Ausreichend Speicherplatz fuer Korpusdateien, Tokenizer-Artefakte und Checkpoints

### Build

```bash
cargo build --release
