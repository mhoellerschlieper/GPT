// utils.rs
// ============================================================================
// Autor:   Marcus Schlieper (ExpChat.ai)
// Hinweis: Config, Tokenizer, Dataset-Lader, Sequenz-Chunking.
// ============================================================================

#![forbid(unsafe_code)]

use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

// ---------------- Config ----------------

pub const MAX_SEQ_LEN_CANONICAL: usize = 256;

// Backward compatible alias: use only if other modules still reference MAX_SEQ_LEN.
pub const MAX_SEQ_LEN: usize = MAX_SEQ_LEN_CANONICAL;

pub const EMBEDDING_DIM: usize = 256;
pub const HIDDEN_DIM: usize = 1024;
pub const HEADS: usize = 8;
pub const DROPOUT: f32 = 0.1;

pub const LEARN_RATE_PRETRAIN: f32 = 1e-3;
pub const LEARN_RATE_TRAIN: f32 = 5e-4;

pub const S_EOS: &str = "</s>";
pub const S_UNK: &str = "<unk>";

// ---------------- Dataset ----------------

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(clippy::upper_case_acronyms)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(pretraining_data_path: String, chat_training_data_path: String, type_of_data: DatasetType) -> Self {
        let (pretraining_data, chat_training_data) = match type_of_data {
            DatasetType::CSV => (get_data_from_csv(pretraining_data_path), get_data_from_csv(chat_training_data_path)),
            DatasetType::JSON => (get_data_from_json(pretraining_data_path), get_data_from_json(chat_training_data_path)),
        };
        Dataset { pretraining_data, chat_training_data }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    let data_json = fs::read_to_string(path).expect("Failed to read data file");
    let data: Vec<String> = serde_json::from_str(&data_json).expect("Failed to parse data file");
    data
}

pub fn get_data_from_csv(path: String) -> Vec<String> {
    let file_path = Path::new(&path);
    let file = fs::File::open(file_path)
        .unwrap_or_else(|e| panic!("CSV-Datei {:?} konnte nicht geoeffnet werden: {}", file_path, e));

    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut v_data: Vec<String> = Vec::new();

    for (_i, result) in rdr.records().enumerate() {
        let record = result.expect("Fehler beim Lesen eines CSV-Datensatzes");
        let mut s_line: String = record.iter().collect::<Vec<&str>>().join(",");
        s_line = s_line.trim_end().to_string();
        if !s_line.ends_with(S_EOS) {
            s_line.push_str(S_EOS);
        }
        v_data.push(s_line);
    }

    v_data
}

// ---------------- Tokenizer (Byte-Level BPE) ----------------

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{Read, Write, BufWriter, BufReader},
};

use bincode::{Decode, Encode};



#[derive(Clone, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Tokenizer {
    #[bincode(with_serde)]
    pub(crate) encode: HashMap<String, usize>,
    #[bincode(with_serde)]
    pub(crate) decode: HashMap<usize, String>,
    #[bincode(with_serde)]
    pub(crate) words: Vec<String>,
    merges: Vec<(String, String)>,
    #[serde(skip)]
    #[bincode(with_serde)]
    fast_lookup: HashMap<(String, String), String>,
}

impl Tokenizer {
    pub fn new_byte_level() -> Self {
        let mut encode: HashMap<String, usize> = HashMap::with_capacity(258);
        let mut decode: HashMap<usize, String> = HashMap::with_capacity(258);
        let mut words: Vec<String> = Vec::with_capacity(258);

        for byte in 0u8..=255 {
            let ch = char::from(byte);
            let tok = ch.to_string();
            let id = byte as usize;
            encode.insert(tok.clone(), id);
            decode.insert(id, tok.clone());
            words.push(tok);
        }
        let id_unk = words.len();
        encode.insert(S_UNK.to_string(), id_unk);
        decode.insert(id_unk, S_UNK.to_string());
        words.push(S_UNK.to_string());

        let id_eos = words.len();
        encode.insert(S_EOS.to_string(), id_eos);
        decode.insert(id_eos, S_EOS.to_string());
        words.push(S_EOS.to_string());

        Self {
            encode,
            decode,
            words,
            merges: Vec::new(),
            fast_lookup: HashMap::new(),
        }
    }

    pub fn vocab_size(&self) -> usize { self.words.len() }

    pub fn encode_token(&self, token: &str) -> Option<usize> {
        self.encode.get(token).copied()
    }

    pub fn decode_id(&self, id: usize) -> Option<&String> {
        self.decode.get(&id)
    }

    pub fn eos_id(&self) -> usize {
        self.encode_token(S_EOS).expect("EOS-Token fehlt")
    }

    pub fn unk_id(&self) -> usize {
        self.encode_token(S_UNK).expect("UNK-Token fehlt")
    }

    fn add_merge_token(&mut self, token: &str) -> usize {
        if let Some(id) = self.encode_token(token) { return id; }
        let new_id = self.words.len();
        self.encode.insert(token.to_string(), new_id);
        self.decode.insert(new_id, token.to_string());
        self.words.push(token.to_string());
        new_id
    }

    pub fn train_bpe(&mut self, texts: &[String], merge_limit: usize) {
        let mut corpus: Vec<Vec<String>> = texts
            .iter()
            .map(|s| {
                let mut v: Vec<String> = s.bytes().map(|b| char::from(b).to_string()).collect();
                v.push(S_EOS.to_string());
                v
            })
            .collect();

        for _ in 0..merge_limit {
            let mut pair_count: HashMap<(String, String), usize> = HashMap::new();
            for seq in &corpus {
                for win in seq.windows(2) {
                    if let [l, r] = win {
                        *pair_count.entry((l.clone(), r.clone())).or_insert(0) += 1;
                    }
                }
            }
            let Some(((l, r), freq)) = pair_count.into_iter().max_by_key(|(_, f)| *f) else {
                break;
            };
            if freq < 2 { break; }

            let merged = format!("{l}{r}");
            self.add_merge_token(&merged);
            self.merges.push((l.clone(), r.clone()));
            self.fast_lookup.insert((l.clone(), r.clone()), merged.clone());

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
    }

    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<String> = text.bytes().map(|b| char::from(b).to_string()).collect();
        tokens.push(S_EOS.to_string());

        for (l, r) in &self.merges {
            let merged = self
                .fast_lookup
                .get(&(l.clone(), r.clone()))
                .expect("merge lookup fehlt");
            let mut i = 0usize;
            while i + 1 < tokens.len() {
                if &tokens[i] == l && &tokens[i + 1] == r {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        let unk_id = self.unk_id();
        tokens.iter().map(|t| self.encode_token(t).unwrap_or(unk_id)).collect()
    }

    pub fn decode_tokens(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(tok) = self.decode_id(id) {
                if tok == S_EOS { break; }
                if tok == S_UNK { continue; }
                for c in tok.chars() {
                    bytes.push(c as u8);
                }
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn save(&self, p_vocab: &str, p_merges: &str) -> std::io::Result<()> {
        let f_vocab = OpenOptions::new().create(true).write(true).truncate(true).open(p_vocab)?;
        let mut wv = BufWriter::with_capacity(8 * 1024 * 1024, f_vocab);
        wv.write_all(&(self.words.len() as u64).to_le_bytes())?;
        for w in &self.words {
            let bytes = w.as_bytes();
            wv.write_all(&(bytes.len() as u32).to_le_bytes())?;
            wv.write_all(bytes)?;
        }
        wv.flush().ok();

        let f_merge = OpenOptions::new().create(true).write(true).truncate(true).open(p_merges)?;
        let mut wm = BufWriter::with_capacity(8 * 1024 * 1024, f_merge);
        wm.write_all(&(self.merges.len() as u64).to_le_bytes())?;
        for (l, r) in &self.merges {
            let bl = l.as_bytes();
            let br = r.as_bytes();
            wm.write_all(&(bl.len() as u32).to_le_bytes())?;
            wm.write_all(bl)?;
            wm.write_all(&(br.len() as u32).to_le_bytes())?;
            wm.write_all(br)?;
        }
        wm.flush().ok();
        Ok(())
    }

    pub fn load(p_vocab: &str, p_merges: &str) -> std::io::Result<Self> {
        let fv = File::open(p_vocab)?;
        let mut rv = BufReader::with_capacity(8 * 1024 * 1024, fv);

        let mut buf64 = [0u8; 8];
        rv.read_exact(&mut buf64)?;
        let count = u64::from_le_bytes(buf64) as usize;

        let mut words: Vec<String> = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            rv.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            rv.read_exact(&mut data)?;
            words.push(String::from_utf8(data)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "UTF-8"))?);
        }
        let mut encode = HashMap::with_capacity(words.len());
        let mut decode = HashMap::with_capacity(words.len());
        for (id, tok) in words.iter().enumerate() {
            encode.insert(tok.clone(), id);
            decode.insert(id, tok.clone());
        }

        let fm = File::open(p_merges)?;
        let mut rm = BufReader::with_capacity(8 * 1024 * 1024, fm);

        rm.read_exact(&mut buf64)?;
        let merges_count = u64::from_le_bytes(buf64) as usize;

        let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_count);
        let mut fast: HashMap<(String, String), String> = HashMap::with_capacity(merges_count);

        for _ in 0..merges_count {
            let mut len_buf = [0u8; 4];
            rm.read_exact(&mut len_buf)?;
            let len_l = u32::from_le_bytes(len_buf) as usize;
            let mut bl = vec![0u8; len_l];
            rm.read_exact(&mut bl)?;
            let l = String::from_utf8(bl)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "UTF-8 l"))?;

            rm.read_exact(&mut len_buf)?;
            let len_r = u32::from_le_bytes(len_buf) as usize;
            let mut br = vec![0u8; len_r];
            rm.read_exact(&mut br)?;
            let r = String::from_utf8(br)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "UTF-8 r"))?;

            fast.insert((l.clone(), r.clone()), format!("{l}{r}"));
            merges.push((l, r));
        }

        Ok(Self { encode, decode, words, merges, fast_lookup: fast })
    }
}

// ---------------- Sequenz-Chunking ----------------

pub fn chunk_sequence(v_tokens: &[usize], i_overlap: usize) -> Vec<Vec<usize>> {
    assert!(i_overlap < MAX_SEQ_LEN, "overlap must be smaller than MAX_SEQ_LEN");

    let mut v_chunks: Vec<Vec<usize>> = Vec::new();
    let mut i_start: usize = 0;

    while i_start < v_tokens.len() {
        let i_end: usize = usize::min(i_start + MAX_SEQ_LEN, v_tokens.len());
        v_chunks.push(v_tokens[i_start..i_end].to_vec());

        if i_end == v_tokens.len() {
            break;
        }
        i_start = i_end.saturating_sub(i_overlap);
    }
    v_chunks
}
