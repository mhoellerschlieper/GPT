// ============================================================================
//  Datei       : tokenizer_bpe.rs – Konsolidierte Fassung
//  Autor       : Marcus Schlieper (ExpChat.ai)
//  Erstellt    : 23.11.2025
//  Zweck       : Byte-Level-BPE-Tokenizer mit integriertem Vokabular
// ============================================================================

use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{Read, Write},
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
    /* -----------------------------------------------------------
     *  Vokabular (ehemals struct Vocab)
     * --------------------------------------------------------- */
    #[bincode(with_serde)] pub(crate) encode: HashMap<String, usize>,
    #[bincode(with_serde)] pub(crate) decode: HashMap<usize, String>,
    #[bincode(with_serde)] pub(crate) words : Vec<String>,

    /* -----------------------------------------------------------
     *  BPE-Merges
     * --------------------------------------------------------- */
    merges      : Vec<(String, String)>,
    fast_lookup : HashMap<(String, String), String>, // (l,r) → "lr"
}

// ---------------------------------------------------------------------------
//  Implementierung
// ---------------------------------------------------------------------------
impl Tokenizer {
    // -------- Konstruktion --------------------------------------------------
    /// Erstellt einen reinen Byte-Level-Tokenizer (256 Zeichen + UNK + EOS).
    pub fn new_byte_level() -> Self {
        let mut encode = HashMap::with_capacity(258);
        let mut decode = HashMap::with_capacity(258);
        let mut words  = Vec::with_capacity(258);

        for byte in 0u8..=255 {
            let tok = (byte as char).to_string();
            let id  = byte as usize;
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
    pub fn vocab_size(&self) -> usize { self.words.len() }

    pub fn encode_token(&self, token: &str) -> Option<usize> {
        self.encode.get(token).copied()
    }

    pub fn decode_id(&self, id: usize) -> Option<&String> {
        self.decode.get(&id)
    }

    /// Fügt bei BPE-Training einen neuen Merge-Token hinzu.
    fn add_merge_token(&mut self, token: &str) -> usize {
        if let Some(id) = self.encode_token(token) { return id; }
        let new_id = self.words.len();
        self.encode.insert(token.to_string(), new_id);
        self.decode.insert(new_id, token.to_string());
        self.words.push(token.to_string());
        new_id
    }

    // -------- BPE-Training --------------------------------------------------
    pub fn train_bpe(&mut self, texts: &[String], merge_limit: usize) {
        // 1) Byte-Sequenzen + EOS
        let mut corpus: Vec<Vec<String>> = texts.iter()
            .map(|s| {
                let mut v: Vec<String> =
                    s.bytes().map(|b| (b as char).to_string()).collect();
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
            let Some(((l, r), freq)) = pair_count.into_iter().max_by_key(|(_, f)| *f) else { break };
            if freq < 2 { break; }

            // Merge-Token anlegen
            let merged = format!("{l}{r}");
            self.add_merge_token(&merged);
            self.merges.push((l.clone(), r.clone()));
            self.fast_lookup.insert((l.clone(), r.clone()), merged.clone());

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
        let mut tokens: Vec<String> =
            text.bytes().map(|b| (b as char).to_string()).collect();
        tokens.push(S_EOS.to_string());
        // Greedy-BPE
        for (l, r) in &self.merges {
            let key = (l.clone(), r.clone());
            let merged = &self.fast_lookup[&key];
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
        let unk_id = self.encode_token(S_UNK).expect("UNK fehlt");
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
                if tok == S_EOS { break; }
                tok.chars().for_each(|c| bytes.push(c as u8));
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    // -------- Persistenz ----------------------------------------------------
    pub fn save(&self, p_vocab: &str, p_merges: &str) -> std::io::Result<()> {
        /* Binär: words */
        let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(p_vocab)?;
        f.write_all(&(self.words.len() as u64).to_le_bytes())?;
        for w in &self.words {
            let bytes = w.as_bytes();
            f.write_all(&(bytes.len() as u32).to_le_bytes())?;
            f.write_all(bytes)?;
        }
        /* Text: merges */
        let mut m = OpenOptions::new().create(true).write(true).truncate(true).open(p_merges)?;
        for (l, r) in &self.merges {
            writeln!(m, "{l} {r}")?;
        }
        Ok(())
    }

    pub fn load(p_vocab: &str, p_merges: &str) -> std::io::Result<Self> {
        /* Binär lesen */
        let mut f = File::open(p_vocab)?;
        let mut buf64 = [0u8; 8];
        f.read_exact(&mut buf64)?;
        let count = u64::from_le_bytes(buf64) as usize;
        let mut words = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            f.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            f.read_exact(&mut data)?;
            words.push(String::from_utf8(data)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "UTF-8"))?);
        }
        let mut encode = HashMap::with_capacity(words.len());
        let mut decode = HashMap::with_capacity(words.len());
        for (id, tok) in words.iter().enumerate() {
            encode.insert(tok.clone(), id);
            decode.insert(id, tok.clone());
        }

        /* Merges lesen */
        let txt = std::fs::read_to_string(p_merges)?;
        let mut merges = Vec::new();
        let mut fast  = HashMap::new();
        for ln in txt.lines() {
            if let Some((l, r)) = ln.split_once(' ') {
                merges.push((l.to_string(), r.to_string()));
                fast.insert((l.to_string(), r.to_string()), format!("{l}{r}"));
            }
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
}
