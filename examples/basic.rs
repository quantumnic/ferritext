//! Basic ferritext usage example.

use ferritext_core::OcrEngine;

fn main() {
    let engine = OcrEngine::new_random("eng");

    println!("ferritext v{}", OcrEngine::version());

    // Create a simple test image
    let img = image::DynamicImage::new_rgb8(200, 50);

    match engine.ocr_image(&img) {
        Ok(result) => {
            println!("Text: {}", result.text);
            println!("Confidence: {:.2}", result.confidence);
            println!("Words: {}", result.words.len());

            // JSON output
            let json = ferritext_core::output::to_json(&result);
            println!("\nJSON:\n{}", json);

            // hOCR output
            let hocr = ferritext_core::output::to_hocr(&result);
            println!("\nhOCR:\n{}", hocr);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
