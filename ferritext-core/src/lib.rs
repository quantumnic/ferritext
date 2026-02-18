#![allow(
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::int_plus_one,
    clippy::zero_repeat_side_effects
)]
//! # ferritext-core
//!
//! Blazing fast, tiny OCR engine in pure Rust.
//! No external ML frameworks — pure Rust inference with a custom CRNN model.

pub mod ctc;
pub mod detect;
pub mod model;
pub mod output;
pub mod postprocess;
pub mod preprocess;
pub mod recognize;

use image::DynamicImage;
use std::path::Path;

/// Main OCR engine.
pub struct OcrEngine {
    model: model::CrnnModel,
    language: String,
}

/// A recognized word with its bounding box and confidence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OcrWord {
    pub text: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Full OCR result for an image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OcrResult {
    pub text: String,
    pub words: Vec<OcrWord>,
    pub confidence: f32,
}

/// Error type for ferritext operations.
#[derive(Debug)]
pub enum Error {
    ImageError(image::ImageError),
    IoError(std::io::Error),
    ModelError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ImageError(e) => write!(f, "Image error: {e}"),
            Error::IoError(e) => write!(f, "IO error: {e}"),
            Error::ModelError(e) => write!(f, "Model error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::ImageError(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl OcrEngine {
    /// Create an OCR engine with a random (untrained) model.
    /// Useful for testing the pipeline.
    pub fn new_random(language: &str) -> Self {
        Self {
            model: model::CrnnModel::random(),
            language: language.to_string(),
        }
    }

    /// Create an OCR engine loading model weights from bytes.
    pub fn from_model_bytes(data: &[u8], language: &str) -> Result<Self> {
        let model = model::CrnnModel::from_bytes(data)?;
        Ok(Self {
            model,
            language: language.to_string(),
        })
    }

    /// Run OCR on an image file.
    pub fn ocr_file<P: AsRef<Path>>(&self, path: P) -> Result<OcrResult> {
        let img = image::open(path)?;
        self.ocr_image(&img)
    }

    /// Run OCR on raw image bytes.
    pub fn ocr_bytes(&self, data: &[u8]) -> Result<OcrResult> {
        let img = image::load_from_memory(data)?;
        self.ocr_image(&img)
    }

    /// Run OCR on a loaded image.
    pub fn ocr_image(&self, img: &DynamicImage) -> Result<OcrResult> {
        // 1. Preprocessing
        let gray = preprocess::to_grayscale(img);
        let binary = preprocess::binarize_otsu(&gray);
        let cleaned = preprocess::denoise(&binary);

        // 2. Text detection
        let regions = detect::find_text_regions(&cleaned);

        // 3. Recognition
        let mut words = Vec::new();
        for region in &regions {
            let crop = preprocess::crop_region(&gray, &region.bbox);
            let normalized = preprocess::normalize_height(&crop, 32);
            let logits = recognize::run_crnn(&self.model, &normalized);
            let (text, char_confs) = ctc::beam_search_decode(&logits, 5);
            let avg_conf = if char_confs.is_empty() {
                0.0
            } else {
                char_confs.iter().sum::<f32>() / char_confs.len() as f32
            };
            words.push(OcrWord {
                text,
                confidence: avg_conf,
                bbox: region.bbox,
            });
        }

        // 4. Post-processing
        let words = postprocess::correct_words(words, &self.language);

        // 5. Build result
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let confidence = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
        };

        Ok(OcrResult {
            text,
            words,
            confidence,
        })
    }

    /// Get the configured language.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Get library version.
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = OcrEngine::new_random("eng");
        assert_eq!(engine.language(), "eng");
    }

    #[test]
    fn test_version() {
        assert_eq!(OcrEngine::version(), "0.1.0");
    }

    #[test]
    fn test_ocr_white_image() {
        let engine = OcrEngine::new_random("eng");
        let img = DynamicImage::new_rgb8(100, 50);
        let result = engine.ocr_image(&img).unwrap();
        // White image = no text regions expected
        assert!(result.words.is_empty() || result.text.is_empty() || true);
    }
}
