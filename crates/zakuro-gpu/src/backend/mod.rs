//! presentation backends.

#[cfg(feature = "opengl")]
pub mod gl;
#[cfg(feature = "vulkan")]
pub mod vulkan;

/// one screen's pixels, already converted to straight RGBA8.
pub struct ScreenImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

impl ScreenImage<'_> {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.pixels.is_empty()
    }
}

/// where a screen goes in the window, in pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// works out where the two screens go inside a window, keeping the 3DS's
/// aspect ratio and centring the bottom screen under the top one.
pub fn layout(window_width: u32, window_height: u32) -> (Viewport, Viewport) {
    // the console is 400x240 over 320x240, so the combined image is 400x480.
    const TOTAL_WIDTH: f32 = 400.0;
    const TOTAL_HEIGHT: f32 = 480.0;

    let scale = (window_width as f32 / TOTAL_WIDTH).min(window_height as f32 / TOTAL_HEIGHT);
    let offset_x = (window_width as f32 - TOTAL_WIDTH * scale) / 2.0;
    let offset_y = (window_height as f32 - TOTAL_HEIGHT * scale) / 2.0;

    let top = Viewport {
        x: offset_x,
        y: offset_y,
        width: 400.0 * scale,
        height: 240.0 * scale,
    };
    let bottom = Viewport {
        // the bottom screen is narrower, so it is centered under the top one.
        x: offset_x + 40.0 * scale,
        y: offset_y + 240.0 * scale,
        width: 320.0 * scale,
        height: 240.0 * scale,
    };
    (top, bottom)
}

#[derive(Debug, thiserror::Error)]
pub enum PresentError {
    #[error("{0}")]
    Backend(String),
    #[error("the swapchain is out of date and was recreated")]
    OutOfDate,
}

pub trait Presenter {
    fn name(&self) -> &'static str;

    /// draws both screens into the window.
    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
    ) -> Result<(), PresentError>;

    /// the window changed size.
    fn resize(&mut self, width: u32, height: u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_keeps_the_aspect_ratio_and_centers() {
        // a window exactly 400x480 needs no scaling or offset.
        let (top, bottom) = layout(400, 480);
        assert_eq!(top.x, 0.0);
        assert_eq!(top.y, 0.0);
        assert_eq!(top.width, 400.0);
        assert_eq!(bottom.y, 240.0);
        assert_eq!(bottom.x, 40.0);
        assert_eq!(bottom.width, 320.0);

        // doubling both dimensions doubles the scale.
        let (top, _) = layout(800, 960);
        assert_eq!(top.width, 800.0);
        assert_eq!(top.height, 480.0);

        // a window that is too wide letterboxes horizontally.
        let (top, _) = layout(1000, 480);
        assert_eq!(top.width, 400.0);
        assert_eq!(top.x, 300.0);
    }
}
