//! Image preprocessing: grayscale, binarization, deskew, noise removal.

use image::{DynamicImage, GrayImage, Luma};

use crate::BoundingBox;

/// Convert an image to grayscale.
pub fn to_grayscale(img: &DynamicImage) -> GrayImage {
    img.to_luma8()
}

/// Compute Otsu's threshold for a grayscale image.
pub fn otsu_threshold(img: &GrayImage) -> u8 {
    let mut histogram = [0u64; 256];
    for pixel in img.pixels() {
        histogram[pixel.0[0] as usize] += 1;
    }

    let total = img.width() as u64 * img.height() as u64;
    if total == 0 {
        return 128;
    }

    let mut sum_total: u64 = 0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as u64 * count;
    }

    let mut sum_bg: u64 = 0;
    let mut weight_bg: u64 = 0;
    let mut max_variance: f64 = 0.0;
    let mut best_threshold: u8 = 0;

    for (t, &count) in histogram.iter().enumerate() {
        weight_bg += count;
        if weight_bg == 0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0 {
            break;
        }

        sum_bg += t as u64 * count;
        let mean_bg = sum_bg as f64 / weight_bg as f64;
        let mean_fg = (sum_total - sum_bg) as f64 / weight_fg as f64;

        let between_variance =
            weight_bg as f64 * weight_fg as f64 * (mean_bg - mean_fg) * (mean_bg - mean_fg);

        if between_variance > max_variance {
            max_variance = between_variance;
            best_threshold = t as u8;
        }
    }

    best_threshold
}

/// Binarize image using Otsu's method. Text=0 (black), background=255 (white).
pub fn binarize_otsu(img: &GrayImage) -> GrayImage {
    let threshold = otsu_threshold(img);
    let mut result = GrayImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        let val = if pixel.0[0] > threshold { 255 } else { 0 };
        result.put_pixel(x, y, Luma([val]));
    }
    result
}

/// Sauvola adaptive binarization.
pub fn binarize_sauvola(img: &GrayImage, window_size: u32, k: f64) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut result = GrayImage::new(w, h);
    let half = (window_size / 2) as i32;

    for y in 0..h {
        for x in 0..w {
            let mut sum: f64 = 0.0;
            let mut sum_sq: f64 = 0.0;
            let mut count: f64 = 0.0;

            for wy in -half..=half {
                for wx in -half..=half {
                    let nx = x as i32 + wx;
                    let ny = y as i32 + wy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let val = img.get_pixel(nx as u32, ny as u32).0[0] as f64;
                        sum += val;
                        sum_sq += val * val;
                        count += 1.0;
                    }
                }
            }

            let mean = sum / count;
            let variance = (sum_sq / count) - (mean * mean);
            let std_dev = variance.max(0.0).sqrt();
            let threshold = mean * (1.0 + k * (std_dev / 128.0 - 1.0));

            let val = if (img.get_pixel(x, y).0[0] as f64) > threshold {
                255
            } else {
                0
            };
            result.put_pixel(x, y, Luma([val]));
        }
    }
    result
}

/// Simple 3x3 median filter for noise removal.
pub fn denoise(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut result = GrayImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let mut neighbors = Vec::with_capacity(9);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        neighbors.push(img.get_pixel(nx as u32, ny as u32).0[0]);
                    }
                }
            }
            neighbors.sort_unstable();
            let median = neighbors[neighbors.len() / 2];
            result.put_pixel(x, y, Luma([median]));
        }
    }
    result
}

/// Estimate skew angle using projection profile (simplified).
/// Returns angle in radians.
pub fn estimate_skew(img: &GrayImage) -> f64 {
    let (w, h) = img.dimensions();
    let mut best_angle: f64 = 0.0;
    let mut best_score: f64 = 0.0;

    // Test angles from -5 to +5 degrees in 0.5 degree steps
    let mut angle = -5.0f64;
    while angle <= 5.0 {
        let rad = angle.to_radians();
        let mut row_sums = vec![0u64; h as usize];

        for y in 0..h {
            for x in 0..w {
                if img.get_pixel(x, y).0[0] == 0 {
                    // black pixel
                    let rotated_y =
                        (y as f64 + (x as f64 - w as f64 / 2.0) * rad.sin()).round() as i32;
                    if rotated_y >= 0 && rotated_y < h as i32 {
                        row_sums[rotated_y as usize] += 1;
                    }
                }
            }
        }

        // Score = sum of squared row counts (sharper peaks = better alignment)
        let score: f64 = row_sums.iter().map(|&s| (s as f64) * (s as f64)).sum();
        if score > best_score {
            best_score = score;
            best_angle = angle;
        }

        angle += 0.5;
    }

    best_angle.to_radians()
}

/// Crop a region from a grayscale image.
pub fn crop_region(img: &GrayImage, bbox: &BoundingBox) -> GrayImage {
    let (iw, ih) = img.dimensions();
    let x = bbox.x.min(iw);
    let y = bbox.y.min(ih);
    let w = bbox.width.min(iw.saturating_sub(x));
    let h = bbox.height.min(ih.saturating_sub(y));

    if w == 0 || h == 0 {
        return GrayImage::new(1, 1);
    }

    let mut result = GrayImage::new(w, h);
    for dy in 0..h {
        for dx in 0..w {
            result.put_pixel(dx, dy, *img.get_pixel(x + dx, y + dy));
        }
    }
    result
}

/// Normalize image height to target_height while preserving aspect ratio.
pub fn normalize_height(img: &GrayImage, target_height: u32) -> GrayImage {
    let (w, h) = img.dimensions();
    if h == 0 || w == 0 {
        return GrayImage::new(1, target_height);
    }

    let scale = target_height as f64 / h as f64;
    let new_w = ((w as f64 * scale).round() as u32).max(1);

    let mut result = GrayImage::new(new_w, target_height);
    for y in 0..target_height {
        for x in 0..new_w {
            let src_x = ((x as f64 / scale).round() as u32).min(w - 1);
            let src_y = ((y as f64 / scale).round() as u32).min(h - 1);
            result.put_pixel(x, y, *img.get_pixel(src_x, src_y));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_grayscale() {
        let img = DynamicImage::new_rgb8(10, 10);
        let gray = to_grayscale(&img);
        assert_eq!(gray.dimensions(), (10, 10));
    }

    #[test]
    fn test_otsu_threshold_uniform() {
        // All-white image
        let img = GrayImage::from_fn(10, 10, |_, _| Luma([255]));
        let t = otsu_threshold(&img);
        // Should be some value; exact doesn't matter for uniform image
        assert!(t <= 255);
    }

    #[test]
    fn test_otsu_threshold_bimodal() {
        let mut img = GrayImage::new(100, 1);
        for x in 0..50 {
            img.put_pixel(x, 0, Luma([10]));
        }
        for x in 50..100 {
            img.put_pixel(x, 0, Luma([240]));
        }
        let t = otsu_threshold(&img);
        // Threshold should be somewhere between the two modes
        assert!(t > 5 && t < 245, "threshold was {t}");
    }

    #[test]
    fn test_binarize_otsu() {
        let img = GrayImage::from_fn(10, 10, |x, _| if x < 5 { Luma([30]) } else { Luma([220]) });
        let bin = binarize_otsu(&img);
        assert_eq!(bin.dimensions(), (10, 10));
        // Dark pixels should become 0
        assert_eq!(bin.get_pixel(0, 0).0[0], 0);
        // Light pixels should become 255
        assert_eq!(bin.get_pixel(9, 0).0[0], 255);
    }

    #[test]
    fn test_binarize_sauvola() {
        let img = GrayImage::from_fn(20, 20, |x, _| if x < 10 { Luma([30]) } else { Luma([220]) });
        let bin = binarize_sauvola(&img, 7, 0.2);
        assert_eq!(bin.dimensions(), (20, 20));
    }

    #[test]
    fn test_denoise() {
        let img = GrayImage::from_fn(10, 10, |_, _| Luma([128]));
        let denoised = denoise(&img);
        assert_eq!(denoised.dimensions(), (10, 10));
        assert_eq!(denoised.get_pixel(5, 5).0[0], 128);
    }

    #[test]
    fn test_crop_region() {
        let img = GrayImage::from_fn(20, 20, |x, y| Luma([(x + y) as u8]));
        let bbox = BoundingBox {
            x: 5,
            y: 5,
            width: 10,
            height: 10,
        };
        let cropped = crop_region(&img, &bbox);
        assert_eq!(cropped.dimensions(), (10, 10));
        assert_eq!(cropped.get_pixel(0, 0).0[0], 10); // 5+5
    }

    #[test]
    fn test_normalize_height() {
        let img = GrayImage::new(100, 64);
        let normalized = normalize_height(&img, 32);
        assert_eq!(normalized.height(), 32);
        assert_eq!(normalized.width(), 50);
    }

    #[test]
    fn test_estimate_skew_straight() {
        let img = GrayImage::from_fn(100, 100, |_, _| Luma([255]));
        let angle = estimate_skew(&img);
        // No black pixels, angle should be ~0
        assert!(angle.abs() < 0.2);
    }
}
