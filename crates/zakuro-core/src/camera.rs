//! what the console's cameras see, the frontend picks, host webcam, a picture file, or nothing

use std::sync::Arc;

/// a picture, rgb bytes row after row
#[derive(Debug, Clone, PartialEq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// how it fits a camera frame of another shape
    pub fit: Fit,
}

/// how a picture fits a frame of another shape
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// all of it w/ white around, for a QR code pic
    Whole,
    /// frame filled, overflow cut off, for a camera
    Fill,
}

/// where the cameras get pictures from
pub trait Camera: Send {
    /// a title started or stopped capturing, a webcam may only be on meanwhile
    fn capturing(&mut self, on: bool);
    /// latest picture, none while there's none
    fn picture(&mut self) -> Option<Arc<Picture>>;
}

/// a picture that never changes, a file's
pub struct Still(pub Arc<Picture>);

impl Camera for Still {
    fn capturing(&mut self, _on: bool) {}

    fn picture(&mut self) -> Option<Arc<Picture>> {
        Some(self.0.clone())
    }
}

impl Picture {
    /// the picture at width by height, fitted as it says, as rgb triples
    pub fn scaled(&self, width: u32, height: u32) -> Vec<[u8; 3]> {
        let mut out = vec![[255u8; 3]; (width * height) as usize];
        if self.width == 0 || self.height == 0 || self.rgb.len() < (self.width * self.height * 3) as usize {
            return out;
        }
        let (source_width, source_height) = (self.width as f32, self.height as f32);
        // how many picture pixels one frame pixel spans
        let step = match self.fit {
            Fit::Whole => (source_width / width as f32).max(source_height / height as f32),
            Fit::Fill => (source_width / width as f32).min(source_height / height as f32),
        };
        // where the frame's top left lands on the picture, both centered
        let left = (source_width - width as f32 * step) / 2.0;
        let top = (source_height - height as f32 * step) / 2.0;
        let texel = |x: i32, y: i32| {
            let at = ((y as u32 * self.width + x as u32) * 3) as usize;
            [self.rgb[at] as f32, self.rgb[at + 1] as f32, self.rgb[at + 2] as f32]
        };
        for y in 0..height {
            let v = top + (y as f32 + 0.5) * step - 0.5;
            if v < -0.5 || v > source_height - 0.5 {
                continue;
            }
            for x in 0..width {
                let u = left + (x as f32 + 0.5) * step - 0.5;
                if u < -0.5 || u > source_width - 0.5 {
                    continue;
                }
                // blend the four nearest by distance
                let (u, v) = (u.clamp(0.0, source_width - 1.0), v.clamp(0.0, source_height - 1.0));
                let (x0, y0) = (u.floor() as i32, v.floor() as i32);
                let (x1, y1) = ((x0 + 1).min(self.width as i32 - 1), (y0 + 1).min(self.height as i32 - 1));
                let (fx, fy) = (u - x0 as f32, v - y0 as f32);
                let (a, b, c, d) = (texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1));
                let pixel = &mut out[(y * width + x) as usize];
                for channel in 0..3 {
                    let top = a[channel] + (b[channel] - a[channel]) * fx;
                    let bottom = c[channel] + (d[channel] - c[channel]) * fx;
                    pixel[channel] = (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(width: u32, height: u32, fit: Fit) -> Picture {
        let mut rgb = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[x as u8, y as u8, 7]);
            }
        }
        Picture { width, height, rgb, fit }
    }

    /// at its own size a picture comes back untouched
    #[test]
    fn a_picture_at_its_size_is_itself() {
        let picture = picture(4, 3, Fit::Whole);
        let scaled = picture.scaled(4, 3);
        assert_eq!(scaled[0], [0, 0, 7]);
        assert_eq!(scaled[4 * 2 + 3], [3, 2, 7]);
    }

    /// a square pic whole in a wide frame gets white on both sides, fill crops top and bottom instead
    #[test]
    fn whole_leaves_white_around_and_fill_cuts() {
        let whole = picture(2, 2, Fit::Whole).scaled(4, 2);
        assert_eq!(whole[0], [255; 3]);
        assert_eq!(whole[3], [255; 3]);
        assert_eq!(whole[1], [0, 0, 7]);
        let fill = picture(2, 2, Fit::Fill).scaled(4, 2);
        assert!(fill.iter().all(|pixel| *pixel != [255; 3]));
    }
}
