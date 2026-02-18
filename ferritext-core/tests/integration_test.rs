//! Integration tests for the ferritext OCR pipeline.

use ferritext_core::{output, OcrEngine};
use image::{DynamicImage, GrayImage, Luma};

#[test]
fn test_full_pipeline_white_image() {
    let engine = OcrEngine::new_random("eng");
    let img = DynamicImage::new_rgb8(200, 50);
    let result = engine.ocr_image(&img).unwrap();

    let json = output::to_json(&result);
    assert!(json.contains("text"));

    let hocr = output::to_hocr(&result);
    assert!(hocr.contains("ferritext"));
}

#[test]
fn test_full_pipeline_with_black_region() {
    let engine = OcrEngine::new_random("eng");
    let mut gray = GrayImage::from_fn(200, 50, |_, _| Luma([255]));

    // Draw a few small "character-like" blocks
    for y in 20..30 {
        for x in 20..30 {
            gray.put_pixel(x, y, Luma([0]));
        }
        for x in 35..45 {
            gray.put_pixel(x, y, Luma([0]));
        }
    }

    let img = DynamicImage::ImageLuma8(gray);
    let result = engine.ocr_image(&img).unwrap();
    // Should detect at least one region from the black blocks
    assert!(!result.words.is_empty());
}

#[test]
fn test_output_formats() {
    let engine = OcrEngine::new_random("eng");
    let img = DynamicImage::new_rgb8(100, 32);
    let result = engine.ocr_image(&img).unwrap();

    let plain = output::to_plain_text(&result);
    assert!(plain.len() <= result.text.len() + 1);

    let json = output::to_json(&result);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_object());

    let hocr = output::to_hocr(&result);
    assert!(hocr.contains("<?xml"));
}
