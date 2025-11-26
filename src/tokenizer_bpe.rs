// ============================================================================
//  Datei       : tokenizer_bpe.rs – Konsolidierte Fassung (Fix)
//  Autor       : Marcus Schlieper (ExpChat.ai)
//  Erstellt    : 23.11.2025
//  Zweck       : Byte-Level-BPE-Tokenizer mit integriertem Vokabular
//  Änderungen  : 26.11.2025  MS  Fix: Special-Tokens, Typen, save/load, eos_id()
// ============================================================================

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{Read, Write, BufWriter, BufReader},
};


use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
//  Konstante Spezial-Tokens
// ---------------------------------------------------------------------------
const S_EOS: &str = "</s>";
const S_UNK: &str = "<unk>";

// ---------------------------------------------------------------------------
//  T O K E N I Z E R  –  Vokabular + BPE-Engine
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Tokenizer {
    #[bincode(with_serde)]
    pub(crate) encode: std::collections::HashMap<String, usize>,
    #[bincode(with_serde)]
    pub(crate) decode: std::collections::HashMap<usize, String>,
    #[bincode(with_serde)]
    pub(crate) words: Vec<String>,
    merges: Vec<(String, String)>,
    #[serde(skip)]
    #[bincode(with_serde)]
    fast_lookup: std::collections::HashMap<(String, String), String>,
}

// ---------------------------------------------------------------------------
//  Implementierung
// ---------------------------------------------------------------------------
impl Tokenizer {
    // -------- Konstruktion --------------------------------------------------
    /// Erstellt einen reinen Byte-Level-Tokenizer (256 Zeichen + UNK + EOS).
    pub fn new_byte_level() -> Self {
        let mut encode: HashMap<String, usize> = HashMap::with_capacity(258);
        let mut decode: HashMap<usize, String> = HashMap::with_capacity(258);
        let mut words: Vec<String> = Vec::with_capacity(258);

        // 0..=255: einzelne Bytes als 1-Char-Strings
        for byte in 0u8..=255 {
            let ch = char::from(byte);
            let tok = ch.to_string();
            let id = byte as usize;
            encode.insert(tok.clone(), id);
            decode.insert(id, tok.clone());
            words.push(tok);
        }
        // UNK
        let id_unk = words.len();
        encode.insert(S_UNK.to_string(), id_unk);
        decode.insert(id_unk, S_UNK.to_string());
        words.push(S_UNK.to_string());

        // EOS
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

    // -------- Vokabular-Hilfsfunktionen ------------------------------------
    pub fn vocab_size(&self) -> usize {
        self.words.len()
    }

    pub fn encode_token(&self, token: &str) -> Option<usize> {
        self.encode.get(token).copied()
    }

    pub fn decode_id(&self, id: usize) -> Option<&String> {
        self.decode.get(&id)
    }

    pub fn eos_id(&self) -> usize {
        self.encode_token(S_EOS)
            .expect("EOS-Token fehlt im Vokabular")
    }

    pub fn unk_id(&self) -> usize {
        self.encode_token(S_UNK)
            .expect("UNK-Token fehlt im Vokabular")
    }

    /// Fügt bei BPE-Training einen neuen Merge-Token hinzu.
    fn add_merge_token(&mut self, token: &str) -> usize {
        if let Some(id) = self.encode_token(token) {
            return id;
        }
        let new_id = self.words.len();
        self.encode.insert(token.to_string(), new_id);
        self.decode.insert(new_id, token.to_string());
        self.words.push(token.to_string());
        new_id
    }

    // -------- BPE-Training --------------------------------------------------
    pub fn train_bpe(&mut self, texts: &[String], merge_limit: usize) {
        // 1) Byte-Sequenzen + EOS
        let mut corpus: Vec<Vec<String>> = texts
            .iter()
            .map(|s| {
                let mut v: Vec<String> = s.bytes().map(|b| char::from(b).to_string()).collect();
                v.push(S_EOS.to_string());
                v
            })
            .collect();

        // 2) Greedy-Merge-Schleife
        for _ in 0..merge_limit {
            // Häufigkeiten aller Paare
            let mut pair_count: HashMap<(String, String), usize> = HashMap::new();
            for seq in &corpus {
                for win in seq.windows(2) {
                    if let [l, r] = win {
                        *pair_count.entry((l.clone(), r.clone())).or_insert(0) += 1;
                    }
                }
            }
            // Häufigstes Paar bestimmen
            let Some(((l, r), freq)) = pair_count.into_iter().max_by_key(|(_, f)| *f) else {
                break;
            };
            if freq < 2 {
                break;
            }

            // Merge-Token anlegen
            let merged = format!("{l}{r}");
            self.add_merge_token(&merged);
            self.merges.push((l.clone(), r.clone()));
            self.fast_lookup
                .insert((l.clone(), r.clone()), merged.clone());

            // Vorkommen im Korpus ersetzen
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

    // -------- Kodierung -----------------------------------------------------
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        // Byte-Level-Tokenisierung
        let mut tokens: Vec<String> = text.bytes().map(|b| char::from(b).to_string()).collect();
        tokens.push(S_EOS.to_string());

        // Greedy-BPE: Merge-Liste in Reihenfolge anwenden
        for (l, r) in &self.merges {
            let merged = self
                .fast_lookup
                .get(&(l.clone(), r.clone()))
                .expect("Merge-Lookup fehlt");
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

        // Mapping Token → ID (OOV → UNK)
        let unk_id = self.unk_id();
        tokens
            .iter()
            .map(|t| self.encode_token(t).unwrap_or(unk_id))
            .collect()
    }

    // -------- Dekodierung ---------------------------------------------------
    pub fn decode_tokens(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(tok) = self.decode_id(id) {
                if tok == S_EOS {
                    break;
                }
                if tok == S_UNK {
                    // Unbekanntes Token: auslassen (oder b'?')
                    continue;
                }
                for c in tok.chars() {
                    // Byte-Ebene: die BPE-Merges bestehen aus Byte-Zeichen
                    bytes.push(c as u8);
                }
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    // -------- Persistenz ----------------------------------------------------
    // Hinweis: Beide Dateien werden BINÄR gespeichert (sicher gegen Trennzeichen).
    pub fn save(&self, p_vocab: &str, p_merges: &str) -> std::io::Result<()> {
        // Vokabular (binär)
        let f_vocab = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(p_vocab)?;
        let mut wv = BufWriter::with_capacity(8 * 1024 * 1024, f_vocab);
        wv.write_all(&(self.words.len() as u64).to_le_bytes())?;
        for w in &self.words {
            let bytes = w.as_bytes();
            wv.write_all(&(bytes.len() as u32).to_le_bytes())?;
            wv.write_all(bytes)?;
        }
        wv.flush().ok();

        // Merges (binär)
        let f_merge = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(p_merges)?;
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

    // Laden (beide Dateien gepuffert)
pub fn load(p_vocab: &str, p_merges: &str) -> std::io::Result<Self> {
    // Vokabular
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
    let mut encode = std::collections::HashMap::with_capacity(words.len());
    let mut decode = std::collections::HashMap::with_capacity(words.len());
    for (id, tok) in words.iter().enumerate() {
        encode.insert(tok.clone(), id);
        decode.insert(id, tok.clone());
    }

    // Merges
    let fm = File::open(p_merges)?;
    let mut rm = BufReader::with_capacity(8 * 1024 * 1024, fm);

    rm.read_exact(&mut buf64)?;
    let merges_count = u64::from_le_bytes(buf64) as usize;

    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_count);
    let mut fast: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::with_capacity(merges_count);

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

// ---------------------------------------------------------------------------
//  Unit-Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bpe() {
        let texts = vec!["hello hello helios".into(), "hello world".into()];
        let mut tok = Tokenizer::new_byte_level();
        tok.train_bpe(&texts, 50);
        let sample = "hello helios";
        let ids = tok.encode_text(sample);
        let decoded = tok.decode_tokens(&ids);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn eos_unk_ids_exist() {
        let tok = Tokenizer::new_byte_level();
        assert!(tok.eos_id() < tok.vocab_size());
        assert!(tok.unk_id() < tok.vocab_size());
    }
}
