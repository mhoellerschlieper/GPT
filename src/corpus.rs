// corpus.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Corpus pipeline for building training datasets.
//           Implements a conservative end to end pipeline:
//           - fetch: acquire raw sources (http_file currently supported)
//           - prepare: normalize text into prepared lines with source tagging
//           - filter: apply simple quality filters and deduplication
//           - pack: split by source id into train/val to avoid leakage, then
//                   split train into pretrain and main datasets.
//
//           Design goals:
//           - Safe defaults (validation, size limits, ASCII only file formats)
//           - Deterministic file layout under root_dir/corpora
//           - Defensive input validation and error handling
//
// History:
//  - 2026-01-17: Adds corpus pipeline entry points used by CLI.
//  - 2026-01-18: Adds global dedup and train/val split by source id,
//                plus stable source tagging "[src=...]" for downstream training.
// ============================================================================

#![forbid(unsafe_code)]
#![allow(warnings)]

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const S_CORPORA_DIR: &str = "corpora";
const S_RAW_DIR: &str = "raw";
const S_PREPARED_DIR: &str = "prepared";
const S_FILTERED_DIR: &str = "filtered";
const S_MANIFEST_JSON: &str = "manifest.json";

const S_USER_AGENT: &str = "ExpChat.ai CorpusFetcher/1.0";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusSourceSpec {
    pub s_id: String,
    // supported values: "http_file"
    pub s_kind: String,
    // for http_file: url
    pub s_url: String,
    // persisted name inside raw directory
    pub s_filename: String,
    // optional parameters for future extensions
    pub m_params: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusPackSpec {
    pub s_out_pretrain_json: String,
    pub s_out_main_json: String,

    // portion of train lines used for pretraining dataset
    pub d_pretrain_ratio: f32,

    // hard caps and basic length filters on prepared lines
    pub i_max_lines_total: usize,
    pub i_min_line_len: usize,
    pub i_max_line_len: usize,

    // train/val split by source id ratio (group based)
    pub d_val_ratio_sources: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusSpec {
    pub v_sources: Vec<CorpusSourceSpec>,
    pub pack: CorpusPackSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusFetchManifest {
    // Each entry is (source_id, raw_file_path)
    pub v_files: Vec<(String, String)>,
    pub i_created_unix: u64,
}

pub struct Corpus {
    s_root_dir: String,
}

impl Corpus {
    pub fn new(s_root_dir: String) -> Self {
        Self { s_root_dir }
    }

    // ------------------------------------------------------------------------
    // Pipeline: fetch
    // ------------------------------------------------------------------------
    // Fetches all sources to root_dir/corpora/raw.
    // Writes a manifest.json to root_dir/corpora/manifest.json.
    pub fn corpus_fetch(&self, spec: &CorpusSpec) -> Result<CorpusFetchManifest> {
        self.validate_spec(spec)?;

        let p_raw = self.path_corpora().join(S_RAW_DIR);
        fs::create_dir_all(&p_raw)
            .with_context(|| format!("cannot create directory: {}", p_raw.to_string_lossy()))?;

        let mut v_files: Vec<(String, String)> = Vec::new();

        for src in &spec.v_sources {
            let s_id = src.s_id.trim().to_string();
            let s_kind = src.s_kind.trim().to_string();
            let s_url = src.s_url.trim().to_string();
            let s_filename = sanitize_filename_ascii(&src.s_filename)?;

            if s_kind == "http_file" {
                let p_out = p_raw.join(&s_filename);
                self.fetch_http_file(&s_url, &p_out)?;
                v_files.push((s_id, p_out.to_string_lossy().to_string()));
            } else {
                return Err(anyhow!("unsupported source kind: {}", s_kind));
            }
        }

        let manifest = CorpusFetchManifest {
            v_files,
            i_created_unix: unix_now(),
        };

        let p_manifest = self.path_corpora().join(S_MANIFEST_JSON);
        let s_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| anyhow!("manifest json serialization failed: {}", e))?;
        fs::write(&p_manifest, s_json)
            .with_context(|| format!("cannot write manifest: {}", p_manifest.to_string_lossy()))?;

        Ok(manifest)
    }

    // ------------------------------------------------------------------------
    // Pipeline: prepare
    // ------------------------------------------------------------------------
    // Normalizes raw sources into prepared text files per source.
    // Each prepared line is tagged with a stable source prefix:
    //   "[src=<source_id>] <text>"
    // This enables downstream group based train/val split in train.rs.
    pub fn corpus_prepare(&self, spec: &CorpusSpec, manifest: &CorpusFetchManifest) -> Result<()> {
        self.validate_spec(spec)?;

        let p_prepared = self.path_corpora().join(S_PREPARED_DIR);
        fs::create_dir_all(&p_prepared).with_context(|| {
            format!(
                "cannot create directory: {}",
                p_prepared.to_string_lossy()
            )
        })?;

        // Map source_id -> raw path from manifest
        let mut m_raw: HashMap<String, String> = HashMap::new();
        for (s_id, s_path) in &manifest.v_files {
            m_raw.insert(s_id.clone(), s_path.clone());
        }

        for src in &spec.v_sources {
            let s_id = src.s_id.trim().to_string();
            let s_filename = sanitize_filename_ascii(&src.s_filename)?;
            let p_raw = match m_raw.get(&s_id) {
                Some(p) => PathBuf::from(p),
                None => {
                    // Fallback: compute expected raw path
                    self.path_corpora().join(S_RAW_DIR).join(&s_filename)
                }
            };

            let s_raw = fs::read_to_string(&p_raw).with_context(|| {
                format!(
                    "cannot read raw corpus file: {}",
                    p_raw.to_string_lossy()
                )
            })?;

            let mut v_lines: Vec<String> = Vec::new();
            for line in s_raw.lines() {
                let s_norm = normalize_line_ascii_safe(line);
                if s_norm.is_empty() {
                    continue;
                }
                let s_tagged = format!("[src={}] {}", s_id, s_norm);
                v_lines.push(s_tagged);
            }

            let p_out = p_prepared.join(format!("{}_prepared.txt", sanitize_id_ascii(&s_id)?));
            write_lines_ascii(&p_out, &v_lines)?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Pipeline: filter
    // ------------------------------------------------------------------------
    // Applies conservative filters and deduplication.
    // Produces root_dir/corpora/filtered/filtered_all.txt.
    // This file preserves the "[src=...]" prefix for each line.
    pub fn corpus_filter(&self, spec: &CorpusSpec) -> Result<()> {
        self.validate_spec(spec)?;

        let p_prepared = self.path_corpora().join(S_PREPARED_DIR);
        let p_filtered = self.path_corpora().join(S_FILTERED_DIR);

        fs::create_dir_all(&p_filtered).with_context(|| {
            format!(
                "cannot create directory: {}",
                p_filtered.to_string_lossy()
            )
        })?;

        let mut v_all: Vec<String> = Vec::new();

        for src in &spec.v_sources {
            let s_id = sanitize_id_ascii(src.s_id.trim())?;
            let p_in = p_prepared.join(format!("{}_prepared.txt", s_id));
            if !p_in.exists() {
                continue;
            }
            let s_data = fs::read_to_string(&p_in).with_context(|| {
                format!(
                    "cannot read prepared file: {}",
                    p_in.to_string_lossy()
                )
            })?;

            for line in s_data.lines() {
                let s_line = line.trim();
                if s_line.is_empty() {
                    continue;
                }
                // Apply length filters on the textual payload, not the prefix.
                let s_payload = strip_src_prefix(s_line).unwrap_or(s_line).trim();
                let i_len = s_payload.len();
                if i_len < spec.pack.i_min_line_len || i_len > spec.pack.i_max_line_len {
                    continue;
                }
                if contains_suspicious_repetition(s_payload) {
                    continue;
                }
                v_all.push(s_line.to_string());
                if v_all.len() >= spec.pack.i_max_lines_total {
                    break;
                }
            }

            if v_all.len() >= spec.pack.i_max_lines_total {
                break;
            }
        }

        // Global deduplication on payload content to reduce trivial copies.
        let v_dedup = dedup_by_payload(&v_all);

        let p_out = p_filtered.join("filtered_all.txt");
        write_lines_ascii(&p_out, &v_dedup)?;

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Pipeline: pack
    // ------------------------------------------------------------------------
    // Reads filtered_all.txt and produces:
    // - data/<out_pretrain_json> (train subset for pretraining)
    // - data/<out_main_json>     (train subset for main training)
    // Additionally writes:
    // - data/corpora_val_by_source.json (validation samples, group split by source)
    pub fn corpus_pack(&self, spec: &CorpusSpec) -> Result<()> {
        self.validate_spec(spec)?;

        let p_filtered_all = self.path_corpora().join(S_FILTERED_DIR).join("filtered_all.txt");
        let s_data = fs::read_to_string(&p_filtered_all).with_context(|| {
            format!(
                "cannot read filtered file: {}",
                p_filtered_all.to_string_lossy()
            )
        })?;

        let mut v_tagged: Vec<String> = Vec::new();
        for line in s_data.lines() {
            let s_line = line.trim();
            if s_line.is_empty() {
                continue;
            }
            v_tagged.push(s_line.to_string());
        }

        // Split by source id (group split) to avoid leakage.
        let (v_train_payload, v_val_payload) =
            split_train_val_by_source_id_payload(&v_tagged, spec.pack.d_val_ratio_sources);

        let (v_pre, v_main) =
            split_pretrain_main(&v_train_payload, spec.pack.d_pretrain_ratio);

        let p_out_pre = Path::new(&self.s_root_dir).join(&spec.pack.s_out_pretrain_json);
        let p_out_main = Path::new(&self.s_root_dir).join(&spec.pack.s_out_main_json);
        let p_out_val = Path::new(&self.s_root_dir).join("corpora_val_by_source.json");

        write_json_vec_string(&p_out_pre, &v_pre)?;
        write_json_vec_string(&p_out_main, &v_main)?;
        write_json_vec_string(&p_out_val, &v_val_payload)?;

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------------
    fn path_corpora(&self) -> PathBuf {
        Path::new(&self.s_root_dir).join(S_CORPORA_DIR)
    }

    fn validate_spec(&self, spec: &CorpusSpec) -> Result<()> {
        if spec.v_sources.is_empty() {
            return Err(anyhow!("spec.v_sources is empty"));
        }

        if spec.pack.i_max_lines_total < 100 {
            return Err(anyhow!("pack.i_max_lines_total too small"));
        }
        if spec.pack.i_min_line_len < 1 {
            return Err(anyhow!("pack.i_min_line_len invalid"));
        }
        if spec.pack.i_max_line_len < spec.pack.i_min_line_len {
            return Err(anyhow!("pack.i_max_line_len < i_min_line_len"));
        }
        if !(0.0..=1.0).contains(&spec.pack.d_pretrain_ratio) {
            return Err(anyhow!("pack.d_pretrain_ratio out of range"));
        }
        if !(0.0..=0.5).contains(&spec.pack.d_val_ratio_sources) {
            return Err(anyhow!("pack.d_val_ratio_sources out of range"));
        }

        for src in &spec.v_sources {
            let s_id = src.s_id.trim();
            if s_id.is_empty() {
                return Err(anyhow!("source id is empty"));
            }
            sanitize_id_ascii(s_id)?;
            let s_kind = src.s_kind.trim();
            if s_kind != "http_file" {
                return Err(anyhow!("unsupported source kind: {}", s_kind));
            }
            let s_url = src.s_url.trim();
            if !is_valid_http_url_ascii(s_url) {
                return Err(anyhow!("invalid source url: {}", s_url));
            }
            sanitize_filename_ascii(&src.s_filename)?;
        }

        Ok(())
    }

    fn fetch_http_file(&self, s_url: &str, p_out: &Path) -> Result<()> {
        // Uses ureq for a small, synchronous download.
        // If the project does not depend on ureq yet, add it to Cargo.toml:
        // ureq = "2"
        let resp = ureq::AgentBuilder::new()
            .user_agent(S_USER_AGENT)
            .timeout_read(std::time::Duration::from_secs(60))
            .timeout_write(std::time::Duration::from_secs(60))
            .build()
            .get(s_url)
            .call()
            .with_context(|| format!("http get failed: {}", s_url))?;

        if resp.status() < 200 || resp.status() >= 300 {
            return Err(anyhow!("http status {} for {}", resp.status(), s_url));
        }

        let mut reader = resp.into_reader();
        let mut v_buf: Vec<u8> = Vec::new();
        reader
            .read_to_end(&mut v_buf)
            .context("failed to read http response body")?;

        // Defensive size cap: 512 MB
        if v_buf.len() > 512 * 1024 * 1024 {
            return Err(anyhow!("download too large: {} bytes", v_buf.len()));
        }

        fs::create_dir_all(p_out.parent().unwrap_or(Path::new(".")))
            .with_context(|| format!("cannot create parent dir for {}", p_out.to_string_lossy()))?;

        let mut f = fs::File::create(p_out)
            .with_context(|| format!("cannot create file: {}", p_out.to_string_lossy()))?;
        f.write_all(&v_buf)
            .with_context(|| format!("cannot write file: {}", p_out.to_string_lossy()))?;
        Ok(())
    }
}

// ============================================================================
// Helper functions (ASCII-only, defensive)
// ============================================================================

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs()
}

fn is_valid_http_url_ascii(s_url: &str) -> bool {
    if s_url.is_empty() {
        return false;
    }
    if !s_url.is_ascii() {
        return false;
    }
    s_url.starts_with("http://") || s_url.starts_with("https://")
}

fn sanitize_id_ascii(s_id: &str) -> Result<String> {
    // Allows: a-z A-Z 0-9 _ - .
    // Rejects: spaces, path separators, non ASCII
    if s_id.is_empty() || !s_id.is_ascii() {
        return Err(anyhow!("invalid id (empty or non-ascii)"));
    }
    for b in s_id.as_bytes() {
        let c = *b as char;
        let ok = c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.';
        if !ok {
            return Err(anyhow!("invalid id character: {}", c));
        }
    }
    Ok(s_id.to_string())
}

fn sanitize_filename_ascii(s_name: &str) -> Result<String> {
    // Very conservative: disallow path separators and parent references.
    let s = s_name.trim();
    if s.is_empty() || !s.is_ascii() {
        return Err(anyhow!("invalid filename (empty or non-ascii)"));
    }
    if s.contains('/') || s.contains('\\') {
        return Err(anyhow!("invalid filename: contains path separator"));
    }
    if s.contains("..") {
        return Err(anyhow!("invalid filename: contains parent reference"));
    }
    // Allow typical filename characters
    for b in s.as_bytes() {
        let c = *b as char;
        let ok = c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '+';
        if !ok {
            return Err(anyhow!("invalid filename character: {}", c));
        }
    }
    Ok(s.to_string())
}

fn normalize_line_ascii_safe(s_in: &str) -> String {
    // Conservative normalization:
    // - trim
    // - convert tabs to spaces
    // - remove ASCII control chars except LF/CR/TAB (CR/LF are not expected inside lines)
    // - collapse repeated spaces to a single space
    let mut v: Vec<u8> = Vec::with_capacity(s_in.len());
    let mut b_prev_space = false;

    for b in s_in.as_bytes() {
        if *b == b'\t' {
            if !b_prev_space {
                v.push(b' ');
                b_prev_space = true;
            }
            continue;
        }
        if *b < 0x20 || *b == 0x7F {
            // drop control chars
            continue;
        }
        if *b == b' ' {
            if b_prev_space {
                continue;
            }
            b_prev_space = true;
            v.push(b' ');
            continue;
        }
        b_prev_space = false;
        v.push(*b);
    }

    String::from_utf8_lossy(&v).trim().to_string()
}

fn write_lines_ascii(p_out: &Path, v_lines: &[String]) -> Result<()> {
    let mut out = String::new();
    for s in v_lines {
        // Ensure the file itself stays ASCII only for inspection purposes.
        // Non-ASCII bytes are removed; training tokenizer is byte-level anyway.
        let s_ascii: String = s.chars().filter(|c| c.is_ascii()).collect();
        out.push_str(&s_ascii);
        out.push('\n');
    }
    fs::write(p_out, out).with_context(|| format!("cannot write: {}", p_out.to_string_lossy()))?;
    Ok(())
}

fn write_json_vec_string(p_out: &Path, v: &[String]) -> Result<()> {
    let s = serde_json::to_string(v).map_err(|e| anyhow!("json serialization failed: {}", e))?;
    fs::write(p_out, s).with_context(|| format!("cannot write: {}", p_out.to_string_lossy()))?;
    Ok(())
}

fn strip_src_prefix(s_line: &str) -> Option<&str> {
    // Expected prefix: "[src=...]" then whitespace.
    if !s_line.starts_with("[src=") {
        return None;
    }
    let pos = s_line.find("]")?;
    let rest = &s_line[(pos + 1)..];
    Some(rest)
}

fn parse_src_id_from_tagged_line(s_line: &str) -> Option<String> {
    // Parses "[src=<id>]" prefix.
    if !s_line.starts_with("[src=") {
        return None;
    }
    let pos = s_line.find("]")?;
    let head = &s_line[..=pos]; // includes ]
    // head is "[src=...]" (ASCII only by construction)
    let inner = head.strip_prefix("[src=")?.strip_suffix("]")?;
    if inner.is_empty() {
        return None;
    }
    Some(inner.to_string())
}

fn dedup_by_payload(v_tagged_lines: &[String]) -> Vec<String> {
    // Deduplicate by payload, but keep the first seen tagged line.
    // This reduces duplicates even across sources; it is conservative for small corpora.
    let mut set_seen: HashSet<String> = HashSet::new();
    let mut v_out: Vec<String> = Vec::new();

    for s in v_tagged_lines {
        let s_payload = strip_src_prefix(s).unwrap_or(s.as_str()).trim().to_string();
        if set_seen.insert(s_payload) {
            v_out.push(s.clone());
        }
    }
    v_out
}

fn split_train_val_by_source_id_payload(
    v_tagged_lines: &[String],
    d_val_ratio_sources: f32,
) -> (Vec<String>, Vec<String>) {
    // Group split: select a subset of source ids for validation and place all
    // their samples into validation. This reduces leakage between train and val.
    let d = d_val_ratio_sources.clamp(0.0, 0.5);

    let mut v_sources: Vec<String> = Vec::new();
    for s in v_tagged_lines {
        if let Some(sid) = parse_src_id_from_tagged_line(s) {
            v_sources.push(sid);
        } else {
            v_sources.push("default".to_string());
        }
    }

    v_sources.sort();
    v_sources.dedup();

    let i_val_sources = ((v_sources.len() as f32) * d).round() as usize;
    let i_val_sources = i_val_sources.min(v_sources.len());

    let mut set_val: HashSet<String> = HashSet::new();
    for sid in v_sources.into_iter().take(i_val_sources) {
        set_val.insert(sid);
    }

    let mut v_train: Vec<String> = Vec::new();
    let mut v_val: Vec<String> = Vec::new();

    for s in v_tagged_lines {
        let sid = parse_src_id_from_tagged_line(s).unwrap_or_else(|| "default".to_string());
        let payload = strip_src_prefix(s).unwrap_or(s.as_str()).trim().to_string();
        if set_val.contains(&sid) {
            v_val.push(payload);
        } else {
            v_train.push(payload);
        }
    }

    (v_train, v_val)
}

fn split_pretrain_main(v_train_payload: &[String], d_pretrain_ratio: f32) -> (Vec<String>, Vec<String>) {
    println!("d_pretrain_ratio: {d_pretrain_ratio}");
    let d = d_pretrain_ratio.clamp(0.0, 1.0);
    let n = v_train_payload.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let i_pre = ((n as f32) * d).round() as usize;
    let i_pre = i_pre.min(n);
    (v_train_payload[..i_pre].to_vec(), v_train_payload[i_pre..].to_vec())
}

fn contains_suspicious_repetition(s_payload: &str) -> bool {
    // Very conservative heuristic to drop pathological lines:
    // - extremely high fraction of a single character
    // - very long runs of the same character
    if s_payload.is_empty() {
        return true;
    }
    let bytes = s_payload.as_bytes();
    if bytes.len() < 16 {
        return false;
    }

    // Check longest run
    let mut i_run_max: usize = 1;
    let mut i_run: usize = 1;
    for i in 1..bytes.len() {
        if bytes[i] == bytes[i - 1] {
            i_run += 1;
            if i_run > i_run_max {
                i_run_max = i_run;
            }
        } else {
            i_run = 1;
        }
    }
    if i_run_max >= 32 {
        return true;
    }

    // Frequency of most common byte
    let mut m: HashMap<u8, usize> = HashMap::new();
    for b in bytes {
        *m.entry(*b).or_insert(0) += 1;
    }
    let mut i_best = 0usize;
    for (_k, v) in m {
        if v > i_best {
            i_best = v;
        }
    }
    let d_frac = (i_best as f32) / (bytes.len() as f32);
    d_frac >= 0.60
}
