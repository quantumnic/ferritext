//! Post-processing: dictionary-based correction and confidence scoring.

use crate::OcrWord;

/// Apply post-processing corrections to recognized words.
pub fn correct_words(words: Vec<OcrWord>, _language: &str) -> Vec<OcrWord> {
    words
        .into_iter()
        .filter(|w| !w.text.trim().is_empty())
        .map(|mut w| {
            w.text = w.text.trim().to_string();
            w
        })
        .collect()
}

/// Simple edit distance (Levenshtein) between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Compute overall confidence for a set of words.
pub fn aggregate_confidence(words: &[OcrWord]) -> f32 {
    if words.is_empty() {
        return 0.0;
    }
    words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundingBox;

    fn make_word(text: &str, confidence: f32) -> OcrWord {
        OcrWord {
            text: text.to_string(),
            confidence,
            bbox: BoundingBox {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
        }
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_correct_words_filters_empty() {
        let words = vec![make_word("hello", 0.9), make_word("  ", 0.1)];
        let result = correct_words(words, "eng");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello");
    }

    #[test]
    fn test_aggregate_confidence() {
        let words = vec![make_word("a", 0.8), make_word("b", 0.6)];
        let conf = aggregate_confidence(&words);
        assert!((conf - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_confidence_empty() {
        assert_eq!(aggregate_confidence(&[]), 0.0);
    }
}
