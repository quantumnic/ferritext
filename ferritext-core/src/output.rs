//! Output formatters: plain text, JSON, hOCR.

use crate::OcrResult;

/// Format OCR result as plain text.
pub fn to_plain_text(result: &OcrResult) -> String {
    result.text.clone()
}

/// Format OCR result as JSON.
pub fn to_json(result: &OcrResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Format OCR result as hOCR (HTML-based OCR format).
pub fn to_hocr(result: &OcrResult) -> String {
    let mut hocr = String::new();
    hocr.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    hocr.push_str("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n");
    hocr.push_str("  \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n");
    hocr.push_str("<html xmlns=\"http://www.w3.org/1999/xhtml\">\n");
    hocr.push_str("<head>\n");
    hocr.push_str("  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />\n");
    hocr.push_str("  <meta name=\"ocr-system\" content=\"ferritext\" />\n");
    hocr.push_str("</head>\n");
    hocr.push_str("<body>\n");
    hocr.push_str("  <div class=\"ocr_page\">\n");

    for (i, word) in result.words.iter().enumerate() {
        let bbox = &word.bbox;
        let x1 = bbox.x;
        let y1 = bbox.y;
        let x2 = bbox.x + bbox.width;
        let y2 = bbox.y + bbox.height;
        hocr.push_str(&format!(
            "    <span class=\"ocrx_word\" id=\"word_{}\" title=\"bbox {} {} {} {}; x_wconf {}\">{}</span>\n",
            i,
            x1, y1, x2, y2,
            (word.confidence * 100.0) as u32,
            html_escape(&word.text)
        ));
    }

    hocr.push_str("  </div>\n");
    hocr.push_str("</body>\n");
    hocr.push_str("</html>\n");

    hocr
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundingBox, OcrResult, OcrWord};

    fn sample_result() -> OcrResult {
        OcrResult {
            text: "Hello World".to_string(),
            words: vec![
                OcrWord {
                    text: "Hello".to_string(),
                    confidence: 0.95,
                    bbox: BoundingBox {
                        x: 10,
                        y: 10,
                        width: 50,
                        height: 20,
                    },
                },
                OcrWord {
                    text: "World".to_string(),
                    confidence: 0.90,
                    bbox: BoundingBox {
                        x: 70,
                        y: 10,
                        width: 50,
                        height: 20,
                    },
                },
            ],
            confidence: 0.925,
        }
    }

    #[test]
    fn test_plain_text() {
        let result = sample_result();
        assert_eq!(to_plain_text(&result), "Hello World");
    }

    #[test]
    fn test_json_output() {
        let result = sample_result();
        let json = to_json(&result);
        assert!(json.contains("Hello"));
        assert!(json.contains("World"));
        assert!(json.contains("confidence"));
        // Validate it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["text"], "Hello World");
    }

    #[test]
    fn test_hocr_output() {
        let result = sample_result();
        let hocr = to_hocr(&result);
        assert!(hocr.contains("ocr_page"));
        assert!(hocr.contains("ocrx_word"));
        assert!(hocr.contains("Hello"));
        assert!(hocr.contains("bbox 10 10 60 30"));
        assert!(hocr.contains("ferritext"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<b>test</b>"), "&lt;b&gt;test&lt;/b&gt;");
        assert_eq!(html_escape("a&b"), "a&amp;b");
    }
}
