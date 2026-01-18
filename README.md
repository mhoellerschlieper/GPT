# ExpChat.ai LLM (Rust) &ndash; AI Chat Client for SMEs

## Overview

This repository contains an experimental LLM pipeline implemented in Rust, featuring a custom byte-level BPE tokenizer, a Transformer model, and a CLI-driven training and inference environment. The focus is on a pragmatic, comprehensible end-to-end chain&mdash;from corpus preparation to interactive usage&mdash;emphasizing robust defaults, defensive error handling, and reproducible artifacts (tokenizer persistence, checkpoints, dataset packages).

The project targets core use cases relevant to small and medium-sized enterprises (SMEs), in particular chat-based assistance, knowledge management, and serving as a technical foundation for RPA and AI agents. It is a research-adjacent system that is intentionally kept transparent, thereby providing a solid basis for extensions (e.g., retrieval-augmented generation, evaluation frameworks, data governance).

## Feature Set (High Level)

- Transformer LLM in Rust (f32, ndarray) with a modular layer architecture.
- Byte-level BPE tokenizer with reversible byte encoding (lossless round-trips text &harr; tokens).
- CLI with a menu for:
  - Training (two-phase: pretraining plus main training)
  - Inference (interactive) and inference grid
  - Checkpoint save/load
  - Corpus pipeline: fetch, prepare, filter, pack
- Corpus pipeline with conservative quality filters, global deduplication, and leakage-reducing splits by sources (source IDs).
- Training features:
  - stable softmax (log-sum-exp)
  - AdamW optimizer
  - gradient clipping
  - optional GradScaler logic
  - optional loss masking for chat formats (assistant-only)

## Project Structure

- `src/main.rs`
  - CLI entry point, menu, training and corpus workflow.
- `src/train.rs`
  - Training, validation, inference, sampling, stop sequences, sliding-window logic.
- `src/layers.rs`
  - Model layers (embeddings, TransformerBlockV2, attention, FFN, normalization, output projection).
- `src/tokenize.rs`
  - Byte-level BPE tokenizer (reversible), merge training, persistence.
- `src/math.rs`
  - Numerical utilities (AdamW, stable softmax, gradient clipping, GradScaler).
- `src/corpus.rs`
  - Corpus pipeline (fetch/prepare/filter/pack) and dataset package creation.
- `src/utils.rs`
  - Configurations and dataset loader.

## Quickstart

### Requirements

- Rust toolchain (stable)
- Internet access (optional, for corpus fetch)
- Sufficient disk space for corpus files, tokenizer artifacts, and checkpoints

### Build

bash
cargo build --release


### Run

bash
cargo run --release


After startup, a CLI menu appears, providing training, inference, and the corpus pipeline.

## Data and Artifacts

### Expected Data Files

By default, the application uses two JSON files under `data/`:

- `data/pretraining_data_de.json`
- `data/chat_training_data_de.json`

Both files contain JSON arrays of strings.

### Corpus Pipeline

The menu provides a corpus workflow that operates artifact-based under `data/corpora/` (e.g., raw/prepared/filtered). The pipeline then produces the JSON files mentioned above and optionally additional sidecar files for source-based validation.

Important: In this architecture, the tokenizer itself appends the EOS special token at the end of sequences; therefore, datasets should not append a literal `` as part of the text, in order to avoid duplication and EOS confusion.

## Training

### Two-Phase Training

Training is designed as a two-phase process:

1. Phase A: Pretraining  
   - typically without chat-specific loss masking
2. Phase B: Main Training (Chat)  
   - optionally with assistant-only loss masking to optimize answers more directly

The CLI interactively prompts for relevant hyperparameters and provides defaults.

### Loss Masking (Assistant-only)

For chat data, a masking approach is used that includes only tokens after the `Assistant:` marker in the loss, while the preceding context still serves as conditioning. This increases the specificity of training in a chat setting; however, it requires that the training data consistently reflects the marker scheme, or that a fallback is implemented in the code if the marker is absent.

## Inference

### Interactive Mode

The CLI menu provides an interactive mode to enter prompts and generate responses. Inference uses sampling parameters (top-k, top-p, temperature) and can apply stop sequences to prevent unintended spillover into subsequent dialog turns.

### Sliding Window

For longer contexts, a sliding-window strategy is used, conservatively truncating the context when the maximum sequence length is reached in order to keep inference stable even for longer dialogs.

## Configuration and Stability Notes

- `MAX_SEQ_LEN_CANONICAL` defines the canonical maximum sequence length.
- The tokenizer and checkpoint are persisted so that training and inference can be resumed.
- For very small datasets, explicit validation (validation split) and monitoring of the effectively unmasked tokens are recommended to detect degenerate training states early.

## Security and Responsible Use

This project provides a technical foundation for text generation; it does not include a content safety or compliance layer, which is typically required in production deployments (e.g., PII filtering, policy enforcement, audit logging). For production applications, upstream data governance and downstream output control are recommended.

## Roadmap (Indicative)

- Advanced, source-based leakage avoidance (sidecar-based group splits in the trainer)
- Semantic-level deduplication (e.g., MinHash)
- Evaluation suite for chat quality and format compliance
- RAG integration for knowledge management and internet research workflows
- Systematic hyperparameter schedules and more robust optimizer parameterization

## Contact and Company

ExpChat.ai  
The AI chat client for SMEs from Breckerfeld in the Sauerland region.  
RPA, AI agents, AI internet research, AI knowledge management&mdash;we bring AI to SMEs.

Address:  
Epscheider Str21  
58339 Breckerfeld  
Germany

Contact:
- Contact person: Marcus Schlieper
- Email: mschlieper@ylook.de
- Phone: 49 2338 8748862
- Mobile: 49 15115751864

## Citation (APA)

For academic referencing, the following citation format is recommended:

Schlieper, M. (2026). *ExpChat.ai LLM (Rust): Tokenizer, Transformer Training and Corpus Pipeline* (Version 0.1.0) [Software]. ExpChat.ai. https://github.com//

Note: The link should be updated accordingly in the repository.


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


