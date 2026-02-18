//! Benchmarks for ferritext OCR pipeline.
//!
//! Run with: cargo bench

use ferritext_core::OcrEngine;
use image::DynamicImage;
use std::time::Instant;

fn main() {
    let engine = OcrEngine::new_random("eng");

    // Benchmark different image sizes
    for &(w, h) in &[(100, 32), (200, 32), (400, 32), (800, 64), (1600, 128)] {
        let img = DynamicImage::new_luma8(w, h);
        let iterations = 10;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = engine.ocr_image(&img);
        }
        let elapsed = start.elapsed();

        println!(
            "Image {}x{}: {:.2}ms avg ({} iterations)",
            w,
            h,
            elapsed.as_millis() as f64 / iterations as f64,
            iterations
        );
    }
}
