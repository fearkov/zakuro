//! presentation backends.

#[cfg(feature = "opengl")]
pub mod gl;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "vulkan")]
mod vulkan_overlay;

/// one screen's pixels, already converted to straight RGBA8, or where they
/// are on the GPU, for a presenter sharing the renderer's device.
pub struct ScreenImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
    pub gpu: Option<GpuScreen>,
}

impl ScreenImage<'_> {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || (self.pixels.is_empty() && self.gpu.is_none())
    }
}

/// a screen's picture upright in an image of the renderer's, in the general
/// layout, which the Vulkan presenter draws straight from when the two share
/// a device.
#[derive(Debug, Clone, Copy)]
pub struct GpuScreen {
    #[cfg(feature = "vulkan")]
    pub(crate) view: ash::vk::ImageView,
    /// where the screen's pixels lie in the image, as texture coordinates,
    /// the corner and the size, then the corners inset by half a texel,
    /// which filtering keeps within.
    pub(crate) area: [f32; 4],
    pub(crate) bounds: [f32; 4],
}

/// what is drawn over the screens, a user interface, as textured triangles
/// in window pixels.
#[derive(Default)]
pub struct Overlay {
    /// textures to make or change before drawing.
    pub textures: Vec<OverlayTexture>,
    pub meshes: Vec<OverlayMesh>,
    /// textures that go away once this frame is drawn.
    pub free: Vec<u64>,
}

/// pixels for a texture of the overlay.
pub struct OverlayTexture {
    pub id: u64,
    /// where the pixels go in a texture that exists, none to make it anew at
    /// this size.
    pub offset: Option<[u32; 2]>,
    pub size: [u32; 2],
    /// RGBA, premultiplied by alpha, in sRGB.
    pub pixels: Vec<u8>,
    /// filtered when scaled, rather than nearest.
    pub linear: bool,
}

/// one corner of an overlay triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayVertex {
    /// window pixels from the top left.
    pub position: [f32; 2],
    pub uv: [f32; 2],
    /// premultiplied sRGB.
    pub color: [u8; 4],
}

/// triangles of the overlay that share a texture and a clip.
pub struct OverlayMesh {
    pub texture: u64,
    /// the part of the window they may draw in, left, top, right, bottom.
    pub clip: [u32; 4],
    pub vertices: Vec<OverlayVertex>,
    pub indices: Vec<u32>,
}

/// where a screen goes in the window, in pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// how the screens are arranged in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenLayout {
    /// the top screen over the bottom one, as on the console.
    #[default]
    Stacked,
    /// next to each other, the top screen on the left.
    SideBySide,
    /// the top screen alone.
    TopOnly,
    /// the bottom screen alone.
    BottomOnly,
    /// the top screen big, the bottom one small at its lower right.
    LargeTop,
}

/// how the screens' pixels are drawn when scaled to the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenFilter {
    /// blended between pixels.
    #[default]
    Smooth,
    /// whole pixels kept sharp and blended only where two meet, so that a
    /// screen scaled by an odd amount stays crisp without stairs.
    Sharp,
    /// every pixel a square of the nearest one.
    Pixels,
}

impl ScreenFilter {
    /// the number the present shaders know it by.
    pub(crate) fn mode(self) -> i32 {
        match self {
            ScreenFilter::Smooth => 0,
            ScreenFilter::Pixels => 1,
            ScreenFilter::Sharp => 2,
        }
    }
}

impl ScreenLayout {
    /// the size the screens take up together at the console's resolution.
    pub fn size(self) -> (u32, u32) {
        match self {
            ScreenLayout::Stacked => (400, 480),
            ScreenLayout::SideBySide => (720, 240),
            ScreenLayout::TopOnly => (400, 240),
            ScreenLayout::BottomOnly => (320, 240),
            ScreenLayout::LargeTop => (400 + LARGE_TOP_BOTTOM.0, 240),
        }
    }
}

/// the bottom screen's size next to the big top one, half the console's.
const LARGE_TOP_BOTTOM: (u32, u32) = (160, 120);

/// works out where the screens go inside a window, keeping the 3DS's
/// aspect ratio and centring them, scaled by a whole number when integer
/// asks and the window has room. a screen that does not show has no place.
pub fn layout(window_width: u32, window_height: u32, arrangement: ScreenLayout, integer: bool) -> (Option<Viewport>, Option<Viewport>) {
    let (total_width, total_height) = arrangement.size();
    let (total_width, total_height) = (total_width as f32, total_height as f32);

    let mut scale = (window_width as f32 / total_width).min(window_height as f32 / total_height);
    if integer && scale >= 1.0 {
        scale = scale.floor();
    }
    let mut offset_x = (window_width as f32 - total_width * scale) / 2.0;
    let mut offset_y = (window_height as f32 - total_height * scale) / 2.0;
    if integer {
        // whole pixels, or every texel would not get the same size
        offset_x = offset_x.floor();
        offset_y = offset_y.floor();
    }

    let top = (arrangement != ScreenLayout::BottomOnly).then_some(Viewport {
        x: offset_x,
        y: offset_y,
        width: 400.0 * scale,
        height: 240.0 * scale,
    });
    let bottom = match arrangement {
        // the bottom screen is narrower, so it is centered under the top one.
        ScreenLayout::Stacked => Some(Viewport {
            x: offset_x + 40.0 * scale,
            y: offset_y + 240.0 * scale,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::SideBySide => Some(Viewport {
            x: offset_x + 400.0 * scale,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::LargeTop => {
            // half the top screen's scale, a whole number of it when the
            // screens are scaled by whole numbers
            let half = (LARGE_TOP_BOTTOM.0 as f32 / 320.0) * scale;
            let small = if integer && half >= 1.0 { half.floor() } else { half };
            Some(Viewport {
                x: offset_x + 400.0 * scale,
                y: offset_y + 240.0 * scale - 240.0 * small,
                width: 320.0 * small,
                height: 240.0 * small,
            })
        }
        ScreenLayout::TopOnly => None,
        ScreenLayout::BottomOnly => Some(Viewport {
            x: offset_x,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
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

    /// draws both screens into the window, and the overlay over them.
    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError>;

    /// the window changed size.
    fn resize(&mut self, width: u32, height: u32);

    /// how the screens are arranged from the next frame on.
    fn set_layout(&mut self, arrangement: ScreenLayout);

    /// how the screens are filtered, and whether they are scaled by whole
    /// numbers, from the next frame on.
    fn set_scaling(&mut self, filter: ScreenFilter, integer: bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_keeps_the_aspect_ratio_and_centers() {
        // a window exactly 400x480 needs no scaling or offset.
        let (top, bottom) = layout(400, 480, ScreenLayout::Stacked, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!(top.x, 0.0);
        assert_eq!(top.y, 0.0);
        assert_eq!(top.width, 400.0);
        assert_eq!(bottom.y, 240.0);
        assert_eq!(bottom.x, 40.0);
        assert_eq!(bottom.width, 320.0);

        // doubling both dimensions doubles the scale.
        let top = layout(800, 960, ScreenLayout::Stacked, false).0.unwrap();
        assert_eq!(top.width, 800.0);
        assert_eq!(top.height, 480.0);

        // a window that is too wide letterboxes horizontally.
        let top = layout(1000, 480, ScreenLayout::Stacked, false).0.unwrap();
        assert_eq!(top.width, 400.0);
        assert_eq!(top.x, 300.0);
    }

    #[test]
    fn the_other_layouts_place_the_screens_their_way() {
        // side by side, the bottom screen starts where the top one ends
        let (top, bottom) = layout(1440, 480, ScreenLayout::SideBySide, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width), (0.0, 0.0, 800.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 0.0, 640.0, 480.0));

        // the top screen alone fills a window of its shape, and a taller
        // one centers it
        let (top, bottom) = layout(800, 960, ScreenLayout::TopOnly, false);
        assert!(bottom.is_none());
        let top = top.unwrap();
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 240.0, 800.0, 480.0));

        // and so does the bottom screen alone, a wider window centering it
        let (top, bottom) = layout(800, 480, ScreenLayout::BottomOnly, false);
        assert!(top.is_none());
        let bottom = bottom.unwrap();
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (80.0, 0.0, 640.0, 480.0));
    }

    #[test]
    fn the_large_top_screen_takes_most_of_the_window() {
        // the bottom screen at half size sits at the top one's lower right
        let (top, bottom) = layout(1120, 480, ScreenLayout::LargeTop, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 0.0, 800.0, 480.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 240.0, 320.0, 240.0));
    }

    #[test]
    fn integer_scaling_keeps_whole_multiples() {
        // 2.5 times fits, 2 is used, and the rest is border
        let top = layout(1000, 1200, ScreenLayout::Stacked, true).0.unwrap();
        assert_eq!((top.width, top.height), (800.0, 480.0));
        assert_eq!((top.x, top.y), (100.0, 120.0));
        // a window smaller than the console's screens still shows them whole
        let top = layout(200, 240, ScreenLayout::Stacked, true).0.unwrap();
        assert_eq!(top.width, 200.0);
        // an odd window puts the screens on whole pixels
        let (top, bottom) = layout(1001, 961, ScreenLayout::Stacked, true);
        assert_eq!((top.unwrap().x, top.unwrap().y), (100.0, 0.0));
        assert_eq!(bottom.unwrap().x, 180.0);
        // the small bottom screen of the big top layout keeps a whole scale
        let (top, bottom) = layout(1680, 720, ScreenLayout::LargeTop, true);
        assert_eq!(top.unwrap().width, 1200.0);
        assert_eq!((bottom.unwrap().width, bottom.unwrap().height), (320.0, 240.0));
    }
}
