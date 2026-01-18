// tokenize.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Tokenizer implementation with reversible byte mapping and BPE.
//           FIX: Prevent EOS from becoming part of BPE merges by never including
//           EOS in the BPE training corpus. EOS must exist exactly once as a
//           special token and must not be produced by merges.
// History:
//  - 2026-01-16: Initial version for reversible byte mapping tokenizer and BPE.
//  - 2026-01-18: Fixes BPE training: do not append EOS to training sequences,
//                preventing EOS related merges and merge file pollution.
// ============================================================================

#![forbid(unsafe_code)]

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};

pub const S_EOS: &str = "</s>";
pub const S_UNK: &str = "<unk>";

#[derive(Clone, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Tokenizer {
    #[bincode(with_serde)]
    encode: HashMap<String, usize>,
    #[bincode(with_serde)]
    decode: HashMap<usize, String>,
    #[bincode(with_serde)]
    words: Vec<String>,

    // BPE merges, applied in stored order
    merges: Vec<(String, String)>,

    // Reversible byte encoder, GPT2 style
    // byte_to_ch maps raw byte -> printable unicode char
    // ch_to_byte maps printable unicode char -> raw byte
    #[bincode(with_serde)]
    byte_to_ch: Vec<char>,
    #[bincode(with_serde)]
    ch_to_byte: HashMap<char, u8>,

    // Fast lookup for merges
    #[serde(skip)]
    #[bincode(with_serde)]
    fast_merge: HashMap<(String, String), String>,
}

impl Tokenizer {
    pub fn new_byte_bpe_reversible() -> Self {
        let (byte_to_ch, ch_to_byte) = build_gpt2_byte_encoder();

        let mut encode: HashMap<String, usize> = HashMap::new();
        let mut decode: HashMap<usize, String> = HashMap::new();
        let mut words: Vec<String> = Vec::new();

        // Base vocab: 256 "byte tokens" represented as printable unicode chars
        for i_b in 0u16..=255u16 {
            let b = i_b as u8;
            let ch = byte_to_ch[b as usize];
            let s_tok = ch.to_string();
            let i_id = words.len();
            words.push(s_tok.clone());
            encode.insert(s_tok.clone(), i_id);
            decode.insert(i_id, s_tok);
        }

        // Special tokens
        let i_unk = words.len();
        words.push(S_UNK.to_string());
        encode.insert(S_UNK.to_string(), i_unk);
        decode.insert(i_unk, S_UNK.to_string());

        let i_eos = words.len();
        words.push(S_EOS.to_string());
        encode.insert(S_EOS.to_string(), i_eos);
        decode.insert(i_eos, S_EOS.to_string());

        let mut tok = Self {
            encode,
            decode,
            words,
            merges: Vec::new(),
            byte_to_ch,
            ch_to_byte,
            fast_merge: HashMap::new(),
        };
        tok.rebuild_fast_merge();
        tok
    }

    pub fn vocab_size(&self) -> usize {
        self.words.len()
    }

    pub fn eos_id(&self) -> usize {
        self.encode_token(S_EOS).expect("missing EOS token")
    }

    pub fn unk_id(&self) -> usize {
        self.encode_token(S_UNK).expect("missing UNK token")
    }

    pub fn encode_token(&self, s_token: &str) -> Option<usize> {
        self.encode.get(s_token).copied()
    }

    pub fn decode_id(&self, i_id: usize) -> Option<&String> {
        self.decode.get(&i_id)
    }

    fn add_token_if_missing(&mut self, s_token: &str) -> usize {
        if let Some(i_id) = self.encode_token(s_token) {
            return i_id;
        }
        let i_new = self.words.len();
        self.words.push(s_token.to_string());
        self.encode.insert(s_token.to_string(), i_new);
        self.decode.insert(i_new, s_token.to_string());
        i_new
    }

    fn rebuild_fast_merge(&mut self) {
        self.fast_merge.clear();
        for (l, r) in &self.merges {
            self.fast_merge
                .insert((l.clone(), r.clone()), format!("{}{}", l, r));
        }
    }

    // ------------------------------------------------------------------------
    // BPE training
    // ------------------------------------------------------------------------
    // FIX:
    // - Do NOT append EOS to sequences during BPE training.
    // - If EOS is present as literal text in inputs, strip a trailing EOS marker.
    // Rationale:
    // - EOS must remain a special token only.
    // - Including EOS in the merge training allows BPE to merge into tokens that
    //   contain "</s>", polluting tokenizer_merges.txt and breaking eos detection.
    pub fn train_bpe(&mut self, v_texts: &[String], i_merge_limit: usize) {
        let mut corpus: Vec<Vec<String>> = v_texts
            .iter()
            .map(|s| {
                // Defensive: remove literal EOS marker at end if present in text.
                let mut s_work = s.to_string();
                if s_work.ends_with(S_EOS) {
                    let i_new_len = s_work.len().saturating_sub(S_EOS.len());
                    s_work.truncate(i_new_len);
                }

                let v_bytes = s_work.as_bytes();
                let v_tokens: Vec<String> = v_bytes
                    .iter()
                    .map(|&b| self.byte_to_ch[b as usize].to_string())
                    .collect();

                // IMPORTANT: no EOS appended here
                v_tokens
            })
            .collect();

        // If corpus is empty or all sequences are too short, merges stay empty.
        if corpus.is_empty() {
            self.merges.clear();
            self.rebuild_fast_merge();
            return;
        }

        for _ in 0..i_merge_limit {
            let mut pair_count: HashMap<(String, String), usize> = HashMap::new();

            for seq in &corpus {
                if seq.len() < 2 {
                    continue;
                }
                for win in seq.windows(2) {
                    let l = &win[0];
                    let r = &win[1];

                    // Safety: never allow merges that include special tokens
                    if l == S_EOS || r == S_EOS || l == S_UNK || r == S_UNK {
                        continue;
                    }

                    *pair_count.entry((l.clone(), r.clone())).or_insert(0) += 1;
                }
            }

            let mut best_pair: Option<((String, String), usize)> = None;
            for (pair, freq) in pair_count {
                if best_pair.is_none() || freq > best_pair.as_ref().unwrap().1 {
                    best_pair = Some((pair, freq));
                }
            }

            let Some(((l, r), freq)) = best_pair else {
                break;
            };

            // Stop if no meaningful repetition remains
            if freq < 2 {
                break;
            }

            // Safety: never create merged tokens that are special tokens or contain them
            if l == S_EOS || r == S_EOS || l == S_UNK || r == S_UNK {
                continue;
            }

            let merged = format!("{}{}", l, r);

            // Safety: prevent merges that would create the literal EOS token
            if merged == S_EOS || merged == S_UNK {
                continue;
            }

            self.add_token_if_missing(&merged);
            self.merges.push((l.clone(), r.clone()));
            self.fast_merge
                .insert((l.clone(), r.clone()), merged.clone());

            // Apply merge to corpus
            for seq in &mut corpus {
                let mut i = 0usize;
                while i + 1 < seq.len() {
                    if seq[i] == l && seq[i + 1] == r {
                        seq[i] = merged.clone();
                        seq.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Final safety: remove any merge lines that reference special tokens
        self.merges
            .retain(|(l, r)| l != S_EOS && r != S_EOS && l != S_UNK && r != S_UNK);

        self.rebuild_fast_merge();
    }

    // ------------------------------------------------------------------------
    // Encoding and decoding (unchanged)
    // ------------------------------------------------------------------------
    pub fn encode_text(&self, s_text: &str) -> Vec<usize> {
        // Defensive: remove literal EOS marker at end to avoid duplication.
        let mut s_work = s_text.to_string();
        if s_work.ends_with(S_EOS) {
            let i_new_len = s_work.len().saturating_sub(S_EOS.len());
            s_work.truncate(i_new_len);
        }

        let mut v_tokens: Vec<String> = s_work
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_ch[b as usize].to_string())
            .collect();

        // Append EOS special token (exactly once)
        v_tokens.push(S_EOS.to_string());

        let mut v_work = v_tokens;
        for (l, r) in &self.merges {
            let merged = self
                .fast_merge
                .get(&(l.clone(), r.clone()))
                .cloned()
                .unwrap_or_else(|| format!("{}{}", l, r));

            let mut i = 0usize;
            while i + 1 < v_work.len() {
                if v_work[i] == *l && v_work[i + 1] == *r {
                    v_work[i] = merged.clone();
                    v_work.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        let i_unk = self.unk_id();
        v_work
            .iter()
            .map(|t| self.encode_token(t).unwrap_or(i_unk))
            .collect()
    }

    pub fn decode_tokens(&self, v_ids: &[usize]) -> String {
        let mut v_bytes: Vec<u8> = Vec::with_capacity(v_ids.len());

        for &i_id in v_ids {
            let Some(s_tok) = self.decode_id(i_id) else {
                continue;
            };

            if s_tok == S_EOS {
                break;
            }
            if s_tok == S_UNK {
                continue;
            }

            for ch in s_tok.chars() {
                if let Some(&b) = self.ch_to_byte.get(&ch) {
                    v_bytes.push(b);
                } else {
                    continue;
                }
            }
        }

        match String::from_utf8(v_bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).to_string(),
        }
    }

    // Persistence (unchanged)
    pub fn save(&self, s_vocab_path: &str, s_merges_path: &str) -> std::io::Result<()> {
        let f_vocab = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(s_vocab_path)?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, f_vocab);

        w.write_all(b"TOK1")?;
        bincode::encode_into_std_write(self, &mut w, bincode::config::standard())
            .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidData, "bincode encode"))?;
        w.flush()?;

        let f_merges = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(s_merges_path)?;
        let mut wm = BufWriter::with_capacity(4 * 1024 * 1024, f_merges);

        for (l, r) in &self.merges {
            let line = format!("{} {}\n", escape_ascii(l), escape_ascii(r));
            wm.write_all(line.as_bytes())?;
        }
        wm.flush()?;
        Ok(())
    }

    pub fn load(s_vocab_path: &str, s_merges_path: &str) -> std::io::Result<Self> {
        let f_vocab = File::open(s_vocab_path)?;
        let mut r = BufReader::with_capacity(8 * 1024 * 1024, f_vocab);

        let mut hdr = [0u8; 4];
        r.read_exact(&mut hdr)?;
        if &hdr != b"TOK1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid tokenizer header",
            ));
        }

        let mut tok: Tokenizer = bincode::decode_from_std_read(&mut r, bincode::config::standard())
            .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidData, "bincode decode"))?;

        let f_merges = File::open(s_merges_path)?;
        let mut rm = BufReader::new(f_merges);
        let mut s_data = String::new();
        rm.read_to_string(&mut s_data)?;

        tok.merges.clear();
        tok.fast_merge.clear();

        for line in s_data.lines() {
            let s_line = line.trim();
            if s_line.is_empty() {
                continue;
            }
            let mut it = s_line.split_whitespace();
            let Some(l) = it.next() else { continue; };
            let Some(r) = it.next() else { continue; };

            let l = unescape_ascii(l);
            let r = unescape_ascii(r);

            // Safety: never load merges that include special tokens
            if l == S_EOS || r == S_EOS || l == S_UNK || r == S_UNK {
                continue;
            }

            tok.merges.push((l.clone(), r.clone()));
            tok.fast_merge.insert((l, r), String::new());
        }
        tok.rebuild_fast_merge();

        Ok(tok)
    }

    pub fn selftest_roundtrip(&self, s_text: &str) -> bool {
        let v_ids = self.encode_text(s_text);
        let s_back = self.decode_tokens(&v_ids);
        s_back == s_text
    }
}

// -----------------------------------------------------------------------------
// GPT2 like byte encoder mapping
// -----------------------------------------------------------------------------
fn build_gpt2_byte_encoder() -> (Vec<char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    for b in 33u16..=126u16 {
        bs.push(b as u8);
    }
    for b in 161u16..=172u16 {
        bs.push(b as u8);
    }
    for b in 174u16..=255u16 {
        bs.push(b as u8);
    }

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut used: HashSet<u8> = bs.iter().copied().collect();

    let mut i_next: u32 = 256;
    for b in 0u16..=255u16 {
        let bb = b as u8;
        if used.contains(&bb) {
            continue;
        }
        cs.push(i_next);
        bs.push(bb);
        i_next += 1;
    }

    let mut byte_to_ch: Vec<char> = vec!['\0'; 256];
    let mut ch_to_byte: HashMap<char, u8> = HashMap::with_capacity(256);

    for (i, &b) in bs.iter().enumerate() {
        let cp = cs[i];
        let ch = std::char::from_u32(cp).unwrap_or('\u{FFFD}');
        byte_to_ch[b as usize] = ch;
        ch_to_byte.insert(ch, b);
    }

    (byte_to_ch, ch_to_byte)
}

// ASCII escape for merges file (keeps file ASCII only)
fn escape_ascii(s_in: &str) -> String {
    let mut out = String::new();
    for b in s_in.as_bytes() {
        let c = *b as char;
        if c.is_ascii_graphic() {
            out.push(c);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

fn unescape_ascii(s_in: &str) -> String {
    let mut out: Vec<u8> = Vec::new();
    let bytes = s_in.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = bytes[i + 1];
            let h2 = bytes[i + 2];
            let v1 = from_hex(h1);
            let v2 = from_hex(h2);
            if v1 >= 0 && v2 >= 0 {
                out.push(((v1 as u8) << 4) | (v2 as u8));
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn from_hex(b: u8) -> i32 {
    match b {
        b'0'..=b'9' => (b - b'0') as i32,
        b'a'..=b'f' => (b - b'a' + 10) as i32,
        b'A'..=b'F' => (b - b'A' + 10) as i32,
        _ => -1,
    }
}
