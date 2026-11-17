//! what the console's cameras see on the computer,
//! its own camera (only on while a game takes pictures) or a picture from a file, like a QR code's

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use zakuro_core::camera::{Camera, Fit, Picture, Still};

/// no console camera sees finer than VGA, a bigger picture gets shrunk when it's read
const LARGEST: u32 = 1280;

/// a picture from a file, shown whole
pub fn picture(path: &Path) -> Result<Still, String> {
    let image = image::open(path).map_err(|error| format!("could not open {}, {error}", path.display()))?;
    let image = if image.width().max(image.height()) > LARGEST {
        image.resize(LARGEST, LARGEST, image::imageops::FilterType::Triangle)
    } else {
        image
    };
    let rgb = image.to_rgb8();
    Ok(Still(Arc::new(Picture { width: rgb.width(), height: rgb.height(), rgb: rgb.into_raw(), fit: Fit::Whole })))
}

/// the computer's camera, read on a thread of its own while a game captures
#[derive(Default)]
pub struct Webcam {
    latest: Arc<Mutex<Option<Arc<Picture>>>>,
    /// should the thread go on reading, while there's one
    running: Option<Arc<AtomicBool>>,
}

impl Camera for Webcam {
    fn capturing(&mut self, on: bool) {
        match (on, &self.running) {
            (true, None) => {
                let running = Arc::new(AtomicBool::new(true));
                let (latest, keep) = (self.latest.clone(), running.clone());
                let started = std::thread::Builder::new().name("camera".into()).spawn(move || read(&latest, &keep));
                match started {
                    Ok(_) => self.running = Some(running),
                    Err(error) => log::warn!("could not start reading the camera, {error}"),
                }
            }
            (false, Some(running)) => {
                // the thread lets go of the camera after the frame it waits on
                running.store(false, Ordering::Relaxed);
                self.running = None;
                if let Ok(mut latest) = self.latest.lock() {
                    *latest = None;
                }
            }
            _ => {}
        }
    }

    fn picture(&mut self) -> Option<Arc<Picture>> {
        self.latest.lock().ok()?.clone()
    }
}

impl Drop for Webcam {
    fn drop(&mut self) {
        self.capturing(false);
    }
}

#[cfg(any(target_os = "linux", windows, target_os = "macos"))]
fn read(latest: &Mutex<Option<Arc<Picture>>>, running: &AtomicBool) {
    use nokhwa::utils::{CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};

    // the formats there's a way to turn into pictures here, the first ones taking the least work
    const FORMATS: [FrameFormat; 6] =
        [FrameFormat::YUYV, FrameFormat::NV12, FrameFormat::MJPEG, FrameFormat::RAWRGB, FrameFormat::RAWBGR, FrameFormat::GRAY];
    let requested = RequestedFormat::with_formats(RequestedFormatType::None, &FORMATS);
    let mut camera = match nokhwa::Camera::new(CameraIndex::Index(0), requested) {
        Ok(camera) => camera,
        Err(error) => {
            log::warn!("no camera to see with, {error}");
            return;
        }
    };
    // the size nearest the console's VGA, at the rate nearest 30 a second
    if let Ok(formats) = camera.compatible_camera_formats() {
        let best = formats.into_iter().filter(|format| FORMATS.contains(&format.format())).min_by_key(|format| {
            let (width, height) = (format.width() as i64, format.height() as i64);
            let rank = FORMATS.iter().position(|known| *known == format.format()).unwrap_or(FORMATS.len());
            ((width - 640).abs() + (height - 480).abs(), rank, (format.frame_rate() as i64 - 30).abs())
        });
        if let Some(best) = best {
            if let Err(error) = camera.set_camera_requset(RequestedFormat::with_formats(RequestedFormatType::Exact(best), &FORMATS)) {
                log::warn!("the camera would not take {best:?}, {error}");
            }
        }
    }
    if let Err(error) = camera.open_stream() {
        log::warn!("could not turn the camera on, {error}");
        return;
    }
    log::info!("the camera is on, {:?}", camera.camera_format());
    while running.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => {
                let size = frame.resolution();
                match decode(frame.buffer(), frame.source_frame_format(), size.width(), size.height()) {
                    Some(picture) => {
                        if let Ok(mut latest) = latest.lock() {
                            *latest = Some(Arc::new(picture));
                        }
                    }
                    None => log::debug!("a frame of the camera could not be read"),
                }
            }
            Err(error) => {
                log::warn!("the camera stopped, {error}");
                break;
            }
        }
    }
    let _ = camera.stop_stream();
    log::info!("the camera is off");
}

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
fn read(_latest: &Mutex<Option<Arc<Picture>>>, _running: &AtomicBool) {
    log::warn!("there is no way to read a camera on this system");
}

/// a frame of the camera as a picture, from the format it came in
#[cfg(any(target_os = "linux", windows, target_os = "macos"))]
fn decode(bytes: &[u8], format: nokhwa::utils::FrameFormat, width: u32, height: u32) -> Option<Picture> {
    use nokhwa::utils::FrameFormat;
    let pixels = (width * height) as usize;
    let picture = |width: u32, height: u32, rgb: Vec<u8>| Picture { width, height, rgb, fit: Fit::Fill };
    match format {
        FrameFormat::MJPEG => {
            let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).ok()?.to_rgb8();
            Some(picture(image.width(), image.height(), image.into_raw()))
        }
        FrameFormat::YUYV => {
            let mut rgb = Vec::with_capacity(pixels * 3);
            for &[y0, u, y1, v] in bytes.get(..pixels * 2)?.as_chunks::<4>().0 {
                rgb.extend_from_slice(&from_yuv(y0, u, v));
                rgb.extend_from_slice(&from_yuv(y1, u, v));
            }
            Some(picture(width, height, rgb))
        }
        FrameFormat::NV12 => {
            let (luma, chroma) = bytes.get(..pixels * 3 / 2)?.split_at(pixels);
            let mut rgb = Vec::with_capacity(pixels * 3);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let at = (y / 2) * width as usize + (x & !1);
                    rgb.extend_from_slice(&from_yuv(luma[y * width as usize + x], chroma[at], chroma[at + 1]));
                }
            }
            Some(picture(width, height, rgb))
        }
        FrameFormat::RAWRGB => Some(picture(width, height, bytes.get(..pixels * 3)?.to_vec())),
        FrameFormat::RAWBGR => {
            let rgb = bytes.get(..pixels * 3)?.as_chunks::<3>().0.iter().flat_map(|&[b, g, r]| [r, g, b]).collect();
            Some(picture(width, height, rgb))
        }
        FrameFormat::GRAY => {
            let rgb = bytes.get(..pixels)?.iter().flat_map(|&gray| [gray; 3]).collect();
            Some(picture(width, height, rgb))
        }
    }
}

/// red, green and blue of a camera's Y, U and V, ITU R BT.601 w/ Y from 16 to 235, like cameras send it
fn from_yuv(y: u8, u: u8, v: u8) -> [u8; 3] {
    let (y, u, v) = (1.164 * (y as f32 - 16.0), u as f32 - 128.0, v as f32 - 128.0);
    let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    [byte(y + 1.596 * v), byte(y - 0.392 * u - 0.813 * v), byte(y + 2.017 * u)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the camera's black and white come out black and white, and red red
    #[test]
    fn camera_colors_come_out_right() {
        assert_eq!(from_yuv(16, 128, 128), [0, 0, 0]);
        assert_eq!(from_yuv(235, 128, 128), [255, 255, 255]);
        let [r, g, b] = from_yuv(81, 90, 240);
        assert!(r > 250 && g < 5 && b < 5, "{r} {g} {b}");
    }

    /// a YUYV frame's pairs of pixels share their U and V
    #[cfg(any(target_os = "linux", windows, target_os = "macos"))]
    #[test]
    fn a_yuyv_frame_becomes_a_picture() {
        let frame = [235, 128, 16, 128];
        let picture = decode(&frame, nokhwa::utils::FrameFormat::YUYV, 2, 1).unwrap();
        assert_eq!(picture.rgb, [255, 255, 255, 0, 0, 0]);
    }
}
