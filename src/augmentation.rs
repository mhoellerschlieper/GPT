// augmentation.rs
// ============================================================================
// Author:   Marcus Schlieper
// Company:  ExpChat.ai
// Contact:  mschlieper@ylook.de | Tel 49 2338 8748862 | Mobil 49 15115751864
// Address:  Epscheider Str21 58339 Breckerfeld
// Note:     Non deterministic online data augmentation helpers.
//           Implements textspace jitter and window length jitter with strict
//           exclusion rules: do not remove markers "User:" and "Assistant:",
//           and do not emit control characters.
// History:
//  - 2026-01-18: Initial implementation of non deterministic augmentation.
// ============================================================================

#![forbid(unsafe_code)]

use rand::Rng;

#[derive(Clone, Debug)]
pub struct AugmentationConfig {
    // Window length jitter in relative terms. Example: 0.15 means plus minus 15 percent.
    pub d_window_len_jitter: f32,

    // Probability to apply whitespace jitter. Example: 0.08 means 8 percent.
    pub d_textspace_jitter_p: f32,

    // Maximum additional spaces that may be inserted when jitter triggers.
    pub i_textspace_max_extra_spaces: usize,

    // Guard rails for chat markers.
    pub b_protect_markers: bool,
}

impl AugmentationConfig {
    pub fn conservative_default() -> Self {
        Self {
            d_window_len_jitter: 0.15,
            d_textspace_jitter_p: 0.08,
            i_textspace_max_extra_spaces: 2,
            b_protect_markers: true,
        }
    }

    pub fn sanitize(&mut self) {
        if !self.d_window_len_jitter.is_finite() || self.d_window_len_jitter < 0.0 {
            self.d_window_len_jitter = 0.0;
        }
        if !self.d_textspace_jitter_p.is_finite() {
            self.d_textspace_jitter_p = 0.0;
        }
        self.d_textspace_jitter_p = self.d_textspace_jitter_p.clamp(0.0, 0.5);
        self.d_window_len_jitter = self.d_window_len_jitter.clamp(0.0, 0.5);
        self.i_textspace_max_extra_spaces = self.i_textspace_max_extra_spaces.clamp(0, 8);
    }
}

pub fn jitter_window_len_non_deterministic(
    i_base_len: usize,
    i_min_len: usize,
    i_max_len: usize,
    d_jitter: f32,
) -> usize {
    let i_base_len = i_base_len.max(2);
    let i_min_len = i_min_len.max(2);
    let i_max_len = i_max_len.max(i_min_len);

    let d_jitter = if d_jitter.is_finite() { d_jitter } else { 0.0 };
    let d_jitter = d_jitter.clamp(0.0, 0.5);

    if d_jitter <= 0.0 {
        return i_base_len.clamp(i_min_len, i_max_len);
    }

    let mut rng = rand::thread_rng();
    let d_base = i_base_len as f32;

    let d_delta = d_base * d_jitter;
    let d_low = (d_base - d_delta).max(i_min_len as f32);
    let d_high = (d_base + d_delta).min(i_max_len as f32);

    if d_low >= d_high {
        return i_base_len.clamp(i_min_len, i_max_len);
    }

    let d_sample = rng.gen_range(d_low..=d_high);
    (d_sample.round() as usize).clamp(i_min_len, i_max_len)
}

pub fn apply_textspace_jitter_non_deterministic(s_in: &str, cfg: &AugmentationConfig) -> String {
    // Exclusion rules:
    // - Do not remove or corrupt "User:" and "Assistant:" markers.
    // - Do not emit control characters (ASCII < 0x20, except '\n', '\t', '\r').
    // - Only apply whitespace modifications (space clusters, newline clusters).
    //
    // The tokenizer is byte reversible, therefore output must remain valid UTF-8.
    // This routine only alters whitespace and remains conservative.

    if s_in.is_empty() {
        return String::new();
    }

    let mut cfg_local = cfg.clone();
    cfg_local.sanitize();

    let d_p = cfg_local.d_textspace_jitter_p;
    if d_p <= 0.0 {
        return sanitize_no_ctrl_preserve_markers(s_in, cfg_local.b_protect_markers);
    }

    let mut rng = rand::thread_rng();

    // Identify protected spans for markers if requested.
    let v_protected = if cfg_local.b_protect_markers {
        find_marker_spans(s_in)
    } else {
        Vec::new()
    };

    let mut out = String::with_capacity(s_in.len() + 16);

    let mut i = 0usize;
    let b = s_in.as_bytes();

    while i < b.len() {
        // If inside a protected span, copy bytes verbatim.
        if is_in_spans(i, &v_protected) {
            out.push(b[i] as char);
            i += 1;
            continue;
        }

        let c = b[i] as char;

        // Sanitize: never emit disallowed control characters.
        if is_disallowed_ascii_control(b[i]) {
            // Drop disallowed control chars conservatively.
            i += 1;
            continue;
        }

        // Apply jitter to spaces and newlines clusters.
        if c == ' ' {
            let i_start = i;
            while i < b.len() && (b[i] as char) == ' ' {
                i += 1;
            }
            let i_len = i - i_start;

            // With probability p, jitter cluster length slightly.
            if rng.r#gen::<f32>() < d_p {
                let i_extra = if cfg_local.i_textspace_max_extra_spaces == 0 {
                    0
                } else {
                    rng.gen_range(0..=cfg_local.i_textspace_max_extra_spaces)
                };

                // Ensure cluster never becomes zero to avoid accidental token glueing.
                let i_new_len = (i_len + i_extra).max(1).min(i_len + cfg_local.i_textspace_max_extra_spaces);
                for _ in 0..i_new_len {
                    out.push(' ');
                }
            } else {
                for _ in 0..i_len {
                    out.push(' ');
                }
            }
            continue;
        }

        if c == '\n' {
            // Keep newline clusters stable, but allow slight normalization to 1..2.
            let i_start = i;
            while i < b.len() && (b[i] as char) == '\n' {
                i += 1;
            }
            let i_len = i - i_start;

            if rng.r#gen::<f32>() < (d_p * 0.25) {
                let i_new_len = i_len.clamp(1, 2);
                for _ in 0..i_new_len {
                    out.push('\n');
                }
            } else {
                for _ in 0..i_len {
                    out.push('\n');
                }
            }
            continue;
        }

        // Default: copy byte as char (ASCII fast path).
        out.push(c);
        i += 1;
    }

    // Final sanitize pass (defensive) while preserving markers.
    sanitize_no_ctrl_preserve_markers(&out, cfg_local.b_protect_markers)
}

fn sanitize_no_ctrl_preserve_markers(s_in: &str, b_protect_markers: bool) -> String {
    if s_in.is_empty() {
        return String::new();
    }

    let v_protected = if b_protect_markers {
        find_marker_spans(s_in)
    } else {
        Vec::new()
    };

    let mut out = String::with_capacity(s_in.len());
    let b = s_in.as_bytes();

    let mut i = 0usize;
    while i < b.len() {
        if is_in_spans(i, &v_protected) {
            out.push(b[i] as char);
            i += 1;
            continue;
        }

        if is_disallowed_ascii_control(b[i]) {
            i += 1;
            continue;
        }

        out.push(b[i] as char);
        i += 1;
    }

    out
}

fn is_disallowed_ascii_control(b: u8) -> bool {
    // Allowed: '\n' (0x0A), '\r' (0x0D), '\t' (0x09)
    // Disallow all other ASCII control characters < 0x20 and DEL (0x7F).
    if b == b'\n' || b == b'\r' || b == b'\t' {
        return false;
    }
    (b < 0x20) || (b == 0x7F)
}

fn find_marker_spans(s_in: &str) -> Vec<(usize, usize)> {
    // Protect exact ASCII marker occurrences.
    // Spans are byte offsets [start, end).
    let mut v = Vec::new();
    let s_user = "User:";
    let s_assistant = "Assistant:";

    for (s_marker, i_len) in [(s_user, s_user.len()), (s_assistant, s_assistant.len())] {
        let mut i = 0usize;
        while let Some(pos) = s_in[i..].find(s_marker) {
            let i_start = i + pos;
            let i_end = i_start + i_len;
            v.push((i_start, i_end));
            i = i_end;
            if i >= s_in.len() {
                break;
            }
        }
    }

    // Normalize and merge overlaps.
    v.sort_by_key(|x| x.0);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in v {
        if merged.is_empty() {
            merged.push((a, b));
            continue;
        }
        let last = merged.len() - 1;
        if a <= merged[last].1 {
            merged[last].1 = merged[last].1.max(b);
        } else {
            merged.push((a, b));
        }
    }
    merged
}

fn is_in_spans(i_pos: usize, v_spans: &[(usize, usize)]) -> bool {
    for &(a, b) in v_spans {
        if i_pos >= a && i_pos < b {
            return true;
        }
    }
    false
}
