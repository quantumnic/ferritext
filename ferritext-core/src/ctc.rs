//! CTC beam search decoder.

use crate::model::{BLANK_IDX, CHARSET};

/// Beam search CTC decoding.
/// Takes logits [time_steps, num_classes] and beam width.
/// Returns (decoded_text, per-character confidences).
pub fn beam_search_decode(logits: &[Vec<f32>], beam_width: usize) -> (String, Vec<f32>) {
    if logits.is_empty() {
        return (String::new(), Vec::new());
    }

    let num_classes = logits[0].len();
    if num_classes == 0 {
        return (String::new(), Vec::new());
    }

    // Softmax each timestep
    let probs: Vec<Vec<f32>> = logits.iter().map(|row| softmax(row)).collect();

    // Beam search
    let mut beams: Vec<Beam> = vec![Beam {
        sequence: Vec::new(),
        log_prob: 0.0,
        char_probs: Vec::new(),
    }];

    for t in 0..probs.len() {
        let mut candidates: Vec<Beam> = Vec::new();

        for beam in &beams {
            // Option 1: emit blank
            let blank_prob = probs[t][BLANK_IDX].max(1e-30).ln();
            candidates.push(Beam {
                sequence: beam.sequence.clone(),
                log_prob: beam.log_prob + blank_prob,
                char_probs: beam.char_probs.clone(),
            });

            // Option 2: emit each character
            for c in 1..num_classes {
                let char_prob = probs[t][c].max(1e-30).ln();
                let last = beam.sequence.last().copied();

                // CTC: skip duplicate consecutive characters
                if last == Some(c) {
                    continue;
                }

                let mut new_seq = beam.sequence.clone();
                new_seq.push(c);
                let mut new_probs = beam.char_probs.clone();
                new_probs.push(probs[t][c]);

                candidates.push(Beam {
                    sequence: new_seq,
                    log_prob: beam.log_prob + char_prob,
                    char_probs: new_probs,
                });
            }
        }

        // Keep top beam_width
        candidates.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap());
        candidates.truncate(beam_width);
        beams = candidates;
    }

    // Best beam
    let best = &beams[0];
    let charset_chars: Vec<char> = CHARSET.chars().collect();

    let text: String = best
        .sequence
        .iter()
        .filter_map(|&idx| {
            if idx == BLANK_IDX || idx - 1 >= charset_chars.len() {
                None
            } else {
                Some(charset_chars[idx - 1])
            }
        })
        .collect();

    (text, best.char_probs.clone())
}

/// Greedy CTC decoding (simpler, faster).
pub fn greedy_decode(logits: &[Vec<f32>]) -> (String, Vec<f32>) {
    if logits.is_empty() {
        return (String::new(), Vec::new());
    }

    let probs: Vec<Vec<f32>> = logits.iter().map(|row| softmax(row)).collect();
    let charset_chars: Vec<char> = CHARSET.chars().collect();

    let mut result = Vec::new();
    let mut confidences = Vec::new();
    let mut prev = BLANK_IDX;

    for t in 0..probs.len() {
        let (best_idx, &best_prob) = probs[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if best_idx != BLANK_IDX && best_idx != prev {
            if best_idx - 1 < charset_chars.len() {
                result.push(charset_chars[best_idx - 1]);
                confidences.push(best_prob);
            }
        }
        prev = best_idx;
    }

    (result.into_iter().collect(), confidences)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp.iter().map(|&e| e / sum).collect()
    }
}

#[derive(Clone)]
struct Beam {
    sequence: Vec<usize>,
    log_prob: f32,
    char_probs: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_empty_logits() {
        let (text, confs) = beam_search_decode(&[], 5);
        assert!(text.is_empty());
        assert!(confs.is_empty());
    }

    #[test]
    fn test_greedy_decode_blank() {
        // All logits strongly favor blank (index 0)
        let num_classes = crate::model::num_classes();
        let mut row = vec![-10.0; num_classes];
        row[0] = 10.0; // blank
        let logits = vec![row; 5];
        let (text, _) = greedy_decode(&logits);
        assert!(text.is_empty());
    }

    #[test]
    fn test_greedy_decode_single_char() {
        let num_classes = crate::model::num_classes();
        let mut row = vec![-10.0; num_classes];
        row[2] = 10.0; // index 2 = second char in CHARSET (should be '!')
        let logits = vec![row; 3];
        let (text, confs) = greedy_decode(&logits);
        // Should decode to single char (CTC collapses repeats)
        assert_eq!(text.len(), 1);
        assert_eq!(confs.len(), 1);
    }

    #[test]
    fn test_beam_search_decode_single() {
        let num_classes = crate::model::num_classes();
        let mut row = vec![-10.0; num_classes];
        row[2] = 10.0;
        let logits = vec![row; 3];
        let (text, _) = beam_search_decode(&logits, 3);
        assert!(!text.is_empty());
    }
}
