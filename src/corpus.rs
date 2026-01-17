// corpus.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Corpus pipeline object for data acquisition and preparation.
//           Provides four phases:
//           - corpus_fetch:    Download and manifest creation
//           - corpus_prepare:  Extraction and normalization
//           - corpus_filter:   Deduplication and boilerplate removal
//           - corpus_pack:     Final output packaging for pretrain and main train
//
//           IMPORTANT CHANGE:
//           - Removed sha256_file_hex completely due to reproducible stack overflow.
//           - Manifest still records url, local path, bytes (best effort).
//           - s_sha256_hex is set to "disabled" or an error marker.
//
// History:
//  - 2026-01-17: Initial version.
//  - 2026-01-17: Hardened HTTP download (redirect limit, timeouts, size guard).
//  - 2026-01-17: Removes sha256 hashing to prevent stack overflow. Keeps pipeline alive.
// ============================================================================

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Corpus {
    pub s_root_dir: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusSpec {
    pub v_sources: Vec<CorpusSourceSpec>,
    pub pack: CorpusPackSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusSourceSpec {
    // s_kind values:
    // - "http_file"
    // - "huggingface_snapshot"
    pub s_id: String,
    pub s_kind: String,
    pub s_url: String,
    pub s_filename: String,

    // Optional parameters (reserved for future extensions).
    pub m_params: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusPackSpec {
    pub s_out_pretrain_json: String,
    pub s_out_main_json: String,
    pub d_pretrain_ratio: f32,
    pub i_max_lines_total: usize,
    pub i_min_line_len: usize,
    pub i_max_line_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusManifest {
    pub s_manifest_version: String,
    pub s_created_utc: String,
    pub s_root_dir: String,
    pub v_entries: Vec<CorpusManifestEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusManifestEntry {
    pub s_id: String,
    pub s_kind: String,
    pub s_url: String,
    pub s_raw_path: String,

    // Hashing disabled: keep field for backward compatibility of manifest schema.
    pub s_sha256_hex: String,

    // Best effort file size in bytes (0 if unknown or not applicable).
    pub i_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct CorpusPaths {
    pub p_raw_dir: PathBuf,
    pub p_prepared_dir: PathBuf,
    pub p_filtered_dir: PathBuf,
    pub p_packed_dir: PathBuf,
    pub p_manifest_path: PathBuf,
}

impl Corpus {
    pub fn new(s_root_dir: String) -> Self {
        Self { s_root_dir }
    }

    fn paths(&self) -> CorpusPaths {
        let p_root = PathBuf::from(&self.s_root_dir);
        CorpusPaths {
            p_raw_dir: p_root.join("corpora").join("raw"),
            p_prepared_dir: p_root.join("corpora").join("prepared"),
            p_filtered_dir: p_root.join("corpora").join("filtered"),
            p_packed_dir: p_root.join("corpora").join("packed"),
            p_manifest_path: p_root.join("corpora").join("manifests").join("manifest.json"),
        }
    }

    fn ensure_dirs(&self) -> std::io::Result<()> {
        let paths = self.paths();
        fs::create_dir_all(&paths.p_raw_dir)?;
        fs::create_dir_all(&paths.p_prepared_dir)?;
        fs::create_dir_all(&paths.p_filtered_dir)?;
        fs::create_dir_all(&paths.p_packed_dir)?;
        if let Some(p_parent) = paths.p_manifest_path.parent() {
            fs::create_dir_all(p_parent)?;
        }
        Ok(())
    }

    fn utc_now_string_ascii() -> String {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0));
        format!("unix_{}", d.as_secs())
    }

    fn write_json_pretty<T: Serialize>(p: &Path, v: &T) -> std::io::Result<()> {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(p)?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f);
        let s = serde_json::to_string_pretty(v)
            .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidData, "json encode"))?;
        w.write_all(s.as_bytes())?;
        w.flush()?;
        Ok(())
    }

    fn read_to_lines_utf8(p: &Path, i_max_lines: usize) -> std::io::Result<Vec<String>> {
        let f = File::open(p)?;
        let r = BufReader::with_capacity(8 * 1024 * 1024, f);

        let mut v_out: Vec<String> = Vec::new();
        for line in r.lines() {
            let s = line?;
            if !s.is_empty() {
                v_out.push(s);
                if v_out.len() >= i_max_lines {
                    break;
                }
            }
        }
        Ok(v_out)
    }

    fn validate_source_spec_basic(src: &CorpusSourceSpec) -> std::io::Result<()> {
        if src.s_id.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source id empty",
            ));
        }
        if src.s_kind.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source kind empty",
            ));
        }
        if src.s_url.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source url empty",
            ));
        }
        if src.s_filename.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source filename empty",
            ));
        }
        Ok(())
    }

    fn validate_filename_single_component(s_filename: &str) -> std::io::Result<()> {
        let p = Path::new(s_filename);
        if p.components().count() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "source filename must be a single name (no path components)",
            ));
        }
        for c in p.components() {
            match c {
                Component::Normal(_) => {}
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "invalid filename component",
                    ));
                }
            }
        }
        Ok(())
    }

    fn best_effort_file_bytes(p: &Path) -> u64 {
        // Best effort: avoid any complex traversal. For directories return 0.
        if let Ok(md) = fs::metadata(p) {
            if md.is_file() {
                return md.len();
            }
        }
        0u64
    }

    // ------------------------------------------------------------------------
    // corpus_fetch: Download and manifest (NO SHA256)
    // ------------------------------------------------------------------------
    // History:
    //  - 2026-01-17: Removes sha256 hashing due to stack overflow. Keeps pipeline running.
    pub fn corpus_fetch(&self, spec: &CorpusSpec) -> std::io::Result<CorpusManifest> {
        self.ensure_dirs()?;
        let paths = self.paths();

        let mut v_entries: Vec<CorpusManifestEntry> = Vec::new();

        for src in &spec.v_sources {
            Self::validate_source_spec_basic(src)?;

            if src.s_kind == "http_file" {
                Self::validate_filename_single_component(&src.s_filename)?;

                let p_out = paths.p_raw_dir.join(&src.s_id).join(&src.s_filename);
                if let Some(p_parent) = p_out.parent() {
                    fs::create_dir_all(p_parent)?;
                }

                println!(
                    "CORPUS_FETCH: http_file start id={} url={}",
                    src.s_id, src.s_url
                );

                // Download is still strict: if download fails, return Err.
                // This is intentional: without the file, later phases are meaningless.
                self.download_http_file_hardened(&src.s_url, &p_out)?;

                println!(
                    "CORPUS_FETCH: http_file downloaded id={} path={}",
                    src.s_id,
                    p_out.to_string_lossy()
                );

                let i_bytes = Self::best_effort_file_bytes(&p_out);

                v_entries.push(CorpusManifestEntry {
                    s_id: src.s_id.clone(),
                    s_kind: src.s_kind.clone(),
                    s_url: src.s_url.clone(),
                    s_raw_path: p_out.to_string_lossy().to_string(),
                    s_sha256_hex: "disabled".to_string(),
                    i_bytes,
                });
            } else if src.s_kind == "huggingface_snapshot" {
                // Snapshot download can be implemented later.
                // For now, record an entry and fail explicitly to avoid silent misbehavior.
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "huggingface_snapshot not implemented in this build",
                ));
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "unsupported source kind",
                ));
            }
        }

        let manifest = CorpusManifest {
            s_manifest_version: "1".to_string(),
            s_created_utc: Self::utc_now_string_ascii(),
            s_root_dir: self.s_root_dir.clone(),
            v_entries,
        };

        Self::write_json_pretty(&paths.p_manifest_path, &manifest)?;
        Ok(manifest)
    }

    fn download_http_file_hardened(&self, s_url: &str, p_out: &Path) -> std::io::Result<()> {
        if !(s_url.starts_with("http://") || s_url.starts_with("https://")) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "url must be http or https",
            ));
        }

        let i_limit_bytes: u64 = 2_000_000_000;
        let i_max_redirects: usize = 10;
        let d_timeout = Duration::from_secs(90);

        let mut s_current_url = s_url.to_string();

        for i_step in 0..=i_max_redirects {
            let agent = ureq::AgentBuilder::new()
                .timeout_connect(d_timeout)
                .timeout_read(d_timeout)
                .redirects(0)
                .build();

            let resp = agent
                .get(&s_current_url)
                .call()
                .map_err(|_e| std::io::Error::new(std::io::ErrorKind::Other, "http request failed"))?;

            let i_status = resp.status();

            if i_status >= 300 && i_status <= 399 {
                if i_step == i_max_redirects {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "redirect limit exceeded",
                    ));
                }
                let s_loc = resp.header("Location").ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "redirect without location")
                })?;
                let s_loc = s_loc.to_string();
                if !(s_loc.starts_with("http://") || s_loc.starts_with("https://")) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "unsupported relative redirect location",
                    ));
                }
                s_current_url = s_loc;
                continue;
            }

            if !(i_status >= 200 && i_status <= 299) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("http status not ok: {}", i_status),
                ));
            }

            if let Some(s_len) = resp.header("Content-Length") {
                if let Ok(i_len) = s_len.parse::<u64>() {
                    if i_len > i_limit_bytes {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "download too large (content length)",
                        ));
                    }
                }
            }

            let p_tmp = p_out.with_extension("tmp");
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&p_tmp)?;

            let mut reader = resp.into_reader();
            let mut buf = [0u8; 1024 * 64];
            let mut i_total: u64 = 0;

            loop {
                let n = reader
                    .read(&mut buf)
                    .map_err(|_e| std::io::Error::new(std::io::ErrorKind::Other, "http read failed"))?;
                if n == 0 {
                    break;
                }
                i_total += n as u64;
                if i_total > i_limit_bytes {
                    let _ = fs::remove_file(&p_tmp);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "download too large",
                    ));
                }
                f.write_all(&buf[..n])?;
            }

            f.flush()?;
            drop(f);

            fs::rename(&p_tmp, p_out)?;
            return Ok(());
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "unexpected redirect loop termination",
        ))
    }

    // ------------------------------------------------------------------------
    // corpus_prepare
    // ------------------------------------------------------------------------
    pub fn corpus_prepare(&self, spec: &CorpusSpec, manifest: &CorpusManifest) -> std::io::Result<()> {
        self.ensure_dirs()?;
        let paths = self.paths();

        for entry in &manifest.v_entries {
            let p_in = PathBuf::from(&entry.s_raw_path);

            if p_in.is_dir() {
                continue;
            }
            if !p_in.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "raw file missing",
                ));
            }

            let p_out_dir = paths.p_prepared_dir.join(&entry.s_id);
            fs::create_dir_all(&p_out_dir)?;
            let p_out = p_out_dir.join("prepared.txt");

            let v_lines = Self::read_to_lines_utf8(&p_in, spec.pack.i_max_lines_total)?;
            let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&p_out)?);

            let mut i_written: usize = 0;
            for s in v_lines {
                let s_norm = normalize_text_line(&s);
                let s_clean = strip_explicit_eos_markers(&s_norm);

                if !is_line_acceptable(&s_clean, spec.pack.i_min_line_len, spec.pack.i_max_line_len) {
                    continue;
                }

                w.write_all(s_clean.as_bytes())?;
                w.write_all(b"\n")?;
                i_written += 1;

                if i_written >= spec.pack.i_max_lines_total {
                    break;
                }
            }

            w.flush()?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------------
    // corpus_filter
    // ------------------------------------------------------------------------
    pub fn corpus_filter(&self, spec: &CorpusSpec) -> std::io::Result<()> {
        self.ensure_dirs()?;
        let paths = self.paths();

        let mut set_seen: HashSet<String> = HashSet::new();

        let mut v_prepared_files: Vec<PathBuf> = Vec::new();
        for src in &spec.v_sources {
            let p_in = paths.p_prepared_dir.join(&src.s_id).join("prepared.txt");
            if p_in.exists() {
                v_prepared_files.push(p_in);
            }
        }

        fs::create_dir_all(&paths.p_filtered_dir)?;
        let p_out = paths.p_filtered_dir.join("filtered.txt");
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&p_out)?);

        let mut i_written: usize = 0;
        for p_in in v_prepared_files {
            let f = File::open(&p_in)?;
            let r = BufReader::with_capacity(8 * 1024 * 1024, f);

            for line in r.lines() {
                let s = line?;
                let s_norm = normalize_text_line(&s);

                if !is_line_acceptable(&s_norm, spec.pack.i_min_line_len, spec.pack.i_max_line_len) {
                    continue;
                }
                if is_probable_boilerplate(&s_norm) {
                    continue;
                }
                if set_seen.contains(&s_norm) {
                    continue;
                }
                set_seen.insert(s_norm.clone());

                w.write_all(s_norm.as_bytes())?;
                w.write_all(b"\n")?;
                i_written += 1;

                if i_written >= spec.pack.i_max_lines_total {
                    break;
                }
            }

            if i_written >= spec.pack.i_max_lines_total {
                break;
            }
        }

        w.flush()?;
        Ok(())
    }

    // ------------------------------------------------------------------------
    // corpus_pack
    // ------------------------------------------------------------------------
    pub fn corpus_pack(&self, spec: &CorpusSpec) -> std::io::Result<()> {
        self.ensure_dirs()?;
        let paths = self.paths();

        let p_in = paths.p_filtered_dir.join("filtered.txt");
        if !p_in.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "filtered.txt missing, run corpus_filter first",
            ));
        }

        let f = File::open(&p_in)?;
        let r = BufReader::with_capacity(8 * 1024 * 1024, f);

        let mut v_all: Vec<String> = Vec::new();
        for line in r.lines() {
            let s = line?;
            if is_line_acceptable(&s, spec.pack.i_min_line_len, spec.pack.i_max_line_len) {
                v_all.push(s);
            }
            if v_all.len() >= spec.pack.i_max_lines_total {
                break;
            }
        }

        if v_all.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no lines after filtering",
            ));
        }

        let d_ratio = spec.pack.d_pretrain_ratio.clamp(0.0, 1.0);
        let i_pre = ((v_all.len() as f32) * d_ratio).round() as usize;
        let i_pre = i_pre.min(v_all.len());

        let v_pretrain = v_all[..i_pre].to_vec();
        let v_main = v_all[i_pre..].to_vec();

        let p_out_pre = paths.p_packed_dir.join(&spec.pack.s_out_pretrain_json);
        let p_out_main = paths.p_packed_dir.join(&spec.pack.s_out_main_json);

        if let Some(p_parent) = p_out_pre.parent() {
            fs::create_dir_all(p_parent)?;
        }
        if let Some(p_parent) = p_out_main.parent() {
            fs::create_dir_all(p_parent)?;
        }

        Self::write_json_pretty(&p_out_pre, &v_pretrain)?;
        Self::write_json_pretty(&p_out_main, &v_main)?;

        let p_report = paths.p_packed_dir.join("pack_report.txt");
        let mut w = BufWriter::new(File::create(p_report)?);
        writeln!(w, "total_lines={}", v_all.len())?;
        writeln!(w, "pretrain_lines={}", v_pretrain.len())?;
        writeln!(w, "main_lines={}", v_main.len())?;
        w.flush()?;

        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Helper functions (safe, deterministic, ASCII only in code)
// ----------------------------------------------------------------------------

fn normalize_text_line(s_in: &str) -> String {
    let mut out = String::with_capacity(s_in.len());
    let mut b_last_space = false;

    for ch in s_in.chars() {
        if ch.is_control() {
            continue;
        }
        let is_space = ch.is_whitespace();
        if is_space {
            if !b_last_space {
                out.push(' ');
                b_last_space = true;
            }
        } else {
            out.push(ch);
            b_last_space = false;
        }
    }

    out.trim().to_string()
}

fn strip_explicit_eos_markers(s_in: &str) -> String {
    let mut s = s_in.trim().to_string();

    loop {
        let s_trim = s.trim_end().to_string();
        if s_trim.ends_with("</s>") {
            s = s_trim.trim_end_matches("</s>").trim().to_string();
            continue;
        }
        if s_trim.ends_with("<\\\\/s>") {
            s = s_trim.trim_end_matches("<\\\\/s>").trim().to_string();
            continue;
        }
        break;
    }

    s
}

fn is_line_acceptable(s: &str, i_min: usize, i_max: usize) -> bool {
    let n = s.len();
    if n < i_min {
        return false;
    }
    if n > i_max {
        return false;
    }

    let mut i_letters: usize = 0;
    let mut i_total: usize = 0;
    for ch in s.chars() {
        i_total += 1;
        if ch.is_alphabetic() {
            i_letters += 1;
        }
    }
    if i_total == 0 {
        return false;
    }
    let d_ratio = (i_letters as f32) / (i_total as f32);
    d_ratio >= 0.2
}

fn is_probable_boilerplate(s: &str) -> bool {
    let s_l = s.to_lowercase();

    if s_l.contains("cookie")
        && (s_l.contains("privacy") || s_l.contains("consent") || s_l.contains("tracking"))
    {
        return true;
    }

    if s_l.contains("impressum") && s_l.contains("datenschutz") {
        return true;
    }
    if s_l.contains("all rights reserved") {
        return true;
    }

    let mut i_punct: usize = 0;
    let mut i_len: usize = 0;
    for ch in s.chars() {
        i_len += 1;
        if ch.is_ascii_punctuation() {
            i_punct += 1;
        }
    }
    if i_len > 0 {
        let d = (i_punct as f32) / (i_len as f32);
        if d > 0.35 {
            return true;
        }
    }

    false
}
