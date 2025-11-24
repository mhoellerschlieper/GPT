# Temporal Transformer Rust &mdash; Research Codebase

## Overview
This repository provides a reference- and research-oriented implementation of a Transformer model written entirely in **Rust**. The architecture extends classic designs with **continuous time embeddings**, **FlashAttention-like block aggregation**, **mixed-precision computation**, and **gradient checkpointing**.  
The codebase targets experts in machine learning, systems programming, and high-performance computing who are interested in reproducible experiments aimed at increasing the efficiency of large language and sequence models.

---

## Feature Matrix
| Category                         | Capability                                                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| Temporal Modeling                | `Time2Vec` embedding (Kazemi &amp; Mehri, 2019) plus &Delta;t bias in self-attention                                 |
| Self-Attention                   | Multi-head attention with rotary positional encoding and key/value ring buffer for O(1) decoding          |
| Memory &amp; Compute Optimization    | FlashAttention-style block processing (Dao et al., 2022) and gradient checkpointing                       |
| Numerical Precision              | Switchable `FloatDType` trait (f32/bf16/f16)                                                              |
| Regularization                   | Configurable dropout at head, feed-forward, and residual level                                            |
| Modularity                       | Strict separation into Rust modules (`multi_head_attention`, `feed_forward_geglu`, `time2vec`, etc.)      |

---

## Installation
bash
git clone https://github.com/mhoellerschlieper/GPT.git
cd temporal-transformer-rust
cargo build --release  # requires Rust 1.75+ and ndarray with BLAS support

Optional mixed-precision builds with CUDA-enabled BLAS back ends can be activated; detailed instructions are available in `docs/BUILD_GPU.md`.

---

## Quick Example
rust
use ndarray::Array2;
use temporal_transformer_rust::{
    transformer_block_v2::TransformerBlockV2,
    config::DropoutCfg,
};

const EMBED:   usize = 256;
const HIDDEN:  usize = 1024;
const HEADS:   usize = 8;

fn main() {
    let dropout = DropoutCfg { f_attn: 0.1, f_ff: 0.05, f_block: 0.1 };
    let mut block = TransformerBlockV2::new(EMBED, HIDDEN, HEADS, dropout, true); // true = mixed precision
    block.attention.clear_cache();               // new sequence
    let input      = Array2::ones((1, EMBED));
    let timestamp  = 1_700_000_000_f32;          // example UNIX timestamp
    let output     = block.forward(&amp;input, timestamp, 0);
    println!(&quot;Output norm {}&quot;, output.sum());
}


---

## Data and Training Pipeline
1. **Dataset Loader**  
   `dataset.rs` ingests JSON- or CSV-based training sets, appends missing terminator tokens `` and validates structural consistency.
2. **Pre-processing**  
   A CLI tool in `tools/preprocess` segments raw corpora, injects timestamp prefixes, and converts data into binary Bincode shards.
3. **Training**  
   The training launcher (`src/bin/train.rs`) supports **Distributed Data Parallel** via `rayon` or single-GPU training; Slurm scheduler files are located under `scripts/`.

---

## Project Structure (excerpt)

├── src
│   ├── multi_head_attention.rs   # FlashAttention core + RoPE
│   ├── feed_forward_geglu.rs     # GeGLU feed-forward network
│   ├── time2vec.rs               # continuous time encoding
│   ├── transformer_block_v2.rs   # pre-norm block with dropout
│   └── ...
├── examples                      # comprehensive notebooks via pyo3 bindings
├── docs
│   └── BUILD_GPU.md
└── Cargo.toml


---

## Contributors
Pull requests, replication studies, and discussions of alternatives (e.g., Performer kernel, flash decoding) are welcome. Please follow the guidelines in `CONTRIBUTING.md`.

---

## License
The code is released under the **Apache 2.0 License**; see `LICENSE`.

---

## Authorship and Contact
| Name                  | Role            | E-mail                        | Phone               |
|-----------------------|-----------------|------------------------------|---------------------|
| Marcus Schlieper      | Maintainer      | mschlieper@expchat.ai          | +49 2338 8748862    |
| ExpChat.ai, Breckerfeld | Project host   | see above                    | +49 151 15751864    |

---

## Citation
If you use this implementation in academic publications, please cite:

&gt; Schlieper, M. (2025). *Temporal Transformer Rust: A time-aware, memory-efficient transformer implementation in pure Rust* (Version 0.0.1) \[Software\]. GitHub. git clone https://github.com/mhoellerschlieper/GPT.git

---
