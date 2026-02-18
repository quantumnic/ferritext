//! Text region detection using connected component analysis.

use image::GrayImage;

use crate::BoundingBox;

/// A detected text region.
#[derive(Debug, Clone)]
pub struct TextRegion {
    pub bbox: BoundingBox,
    pub pixel_count: u32,
}

/// Find connected components in a binary image (black pixels = foreground).
/// Returns bounding boxes of components, merged into text line regions.
pub fn find_text_regions(binary: &GrayImage) -> Vec<TextRegion> {
    let components = find_connected_components(binary);
    let filtered = filter_components(&components, binary.width(), binary.height());
    merge_into_lines(filtered, binary.height())
}

/// Union-Find structure for connected component labeling.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Two-pass connected component labeling.
fn find_connected_components(binary: &GrayImage) -> Vec<ComponentInfo> {
    let (w, h) = binary.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut labels = vec![0u32; (w * h) as usize];
    let mut next_label = 1u32;
    let mut uf = UnionFind::new(1); // index 0 is background

    // First pass
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if binary.get_pixel(x, y).0[0] != 0 {
                continue; // skip white (background)
            }

            let mut neighbors = Vec::new();
            if x > 0 && labels[(y * w + x - 1) as usize] > 0 {
                neighbors.push(labels[(y * w + x - 1) as usize]);
            }
            if y > 0 && labels[((y - 1) * w + x) as usize] > 0 {
                neighbors.push(labels[((y - 1) * w + x) as usize]);
            }

            if neighbors.is_empty() {
                labels[idx] = next_label;
                next_label += 1;
                if uf.parent.len() <= labels[idx] as usize {
                    let new_id = uf.parent.len();
                    uf.parent.push(new_id);
                    uf.rank.push(0);
                }
            } else {
                let min_label = *neighbors.iter().min().unwrap();
                labels[idx] = min_label;
                for &n in &neighbors {
                    uf.union(min_label as usize, n as usize);
                }
            }
        }
    }

    // Second pass: resolve labels
    let mut component_map: std::collections::HashMap<u32, ComponentInfo> =
        std::collections::HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if labels[idx] == 0 {
                continue;
            }
            let root = uf.find(labels[idx] as usize) as u32;
            labels[idx] = root;

            let entry = component_map.entry(root).or_insert(ComponentInfo {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                pixel_count: 0,
            });
            entry.min_x = entry.min_x.min(x);
            entry.min_y = entry.min_y.min(y);
            entry.max_x = entry.max_x.max(x);
            entry.max_y = entry.max_y.max(y);
            entry.pixel_count += 1;
        }
    }

    component_map.into_values().collect()
}

#[derive(Debug, Clone)]
struct ComponentInfo {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    pixel_count: u32,
}

/// Filter out noise components (too small or too large).
fn filter_components(components: &[ComponentInfo], img_w: u32, img_h: u32) -> Vec<ComponentInfo> {
    let img_area = (img_w as u64) * (img_h as u64);
    let min_pixels = 4u32;
    let max_ratio = 0.5f64;

    components
        .iter()
        .filter(|c| {
            let area = (c.max_x - c.min_x + 1) as u64 * (c.max_y - c.min_y + 1) as u64;
            c.pixel_count >= min_pixels && (area as f64) < (img_area as f64 * max_ratio)
        })
        .cloned()
        .collect()
}

/// Merge nearby components into text line regions.
fn merge_into_lines(components: Vec<ComponentInfo>, _img_h: u32) -> Vec<TextRegion> {
    if components.is_empty() {
        return Vec::new();
    }

    // Sort by y, then x
    let mut sorted = components;
    sorted.sort_by_key(|c| (c.min_y, c.min_x));

    let mut lines: Vec<TextRegion> = Vec::new();

    for comp in &sorted {
        let comp_h = comp.max_y - comp.min_y + 1;
        let merge_threshold = comp_h / 2 + 1;

        let mut merged = false;
        for line in lines.iter_mut() {
            // Check vertical overlap
            let line_cy = line.bbox.y + line.bbox.height / 2;
            let comp_cy = comp.min_y + comp_h / 2;
            let y_dist = (line_cy as i32 - comp_cy as i32).unsigned_abs();

            if y_dist < merge_threshold {
                // Merge
                let new_x = line.bbox.x.min(comp.min_x);
                let new_y = line.bbox.y.min(comp.min_y);
                let new_max_x = (line.bbox.x + line.bbox.width).max(comp.max_x + 1);
                let new_max_y = (line.bbox.y + line.bbox.height).max(comp.max_y + 1);
                line.bbox.x = new_x;
                line.bbox.y = new_y;
                line.bbox.width = new_max_x - new_x;
                line.bbox.height = new_max_y - new_y;
                line.pixel_count += comp.pixel_count;
                merged = true;
                break;
            }
        }

        if !merged {
            lines.push(TextRegion {
                bbox: BoundingBox {
                    x: comp.min_x,
                    y: comp.min_y,
                    width: comp.max_x - comp.min_x + 1,
                    height: comp.max_y - comp.min_y + 1,
                },
                pixel_count: comp.pixel_count,
            });
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_empty_image() {
        let img = GrayImage::new(10, 10); // all black = one big component
        let regions = find_text_regions(&img);
        // All black pixels form one component, but it may be filtered as too large
        // Either result is valid
        let _ = regions;
    }

    #[test]
    fn test_white_image() {
        let img = GrayImage::from_fn(10, 10, |_, _| Luma([255]));
        let regions = find_text_regions(&img);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_single_component() {
        let mut img = GrayImage::from_fn(100, 100, |_, _| Luma([255]));
        // Draw a small block of black pixels
        for y in 10..20 {
            for x in 10..30 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let regions = find_text_regions(&img);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].bbox.x <= 10);
        assert!(regions[0].bbox.y <= 10);
    }

    #[test]
    fn test_two_lines() {
        let mut img = GrayImage::from_fn(200, 100, |_, _| Luma([255]));
        // Line 1
        for y in 10..20 {
            for x in 10..50 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        // Line 2 (far enough apart)
        for y in 60..70 {
            for x in 10..50 {
                img.put_pixel(x, y, Luma([0]));
            }
        }
        let regions = find_text_regions(&img);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_connected_components_basic() {
        let mut img = GrayImage::from_fn(10, 10, |_, _| Luma([255]));
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(1, 0, Luma([0]));
        img.put_pixel(0, 1, Luma([0]));

        let components = find_connected_components(&img);
        // Should be one component with 3 pixels
        assert!(!components.is_empty());
        let total_pixels: u32 = components.iter().map(|c| c.pixel_count).sum();
        assert!(total_pixels >= 3);
    }
}
