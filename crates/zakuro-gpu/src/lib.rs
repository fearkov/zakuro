//! PICA200 GPU emulation.

pub mod backend;
pub mod blend;
#[cfg(feature = "vulkan")]
mod device;
mod pattern;
pub mod proctex;
pub mod fog;
pub mod format;
pub mod lighting;
pub mod pack;
pub mod raster;
pub mod registers;
pub mod renderer;
pub mod shader;
pub mod tev;
pub mod texture;

use format::ColorFormat;
use registers::*;
pub use backend::{layout, GpuScreen, Overlay, OverlayMesh, OverlayTexture, OverlayVertex, PresentError, Presenter, ScreenFilter, ScreenImage, ScreenLayout, Viewport};
pub use renderer::{DrawCall, Renderer, RendererKind, SoftwareRenderer};
#[cfg(feature = "vulkan")]
pub use device::SharedDevice;

/// how the GPU reaches guest memory.
pub trait GpuMemory {
    fn read(&mut self, addr: u32, out: &mut [u8]);
    fn write(&mut self, addr: u32, data: &[u8]);

    /// the bytes read would find at addr, straight from memory when they
    /// sit together there, to look at without copying them.
    fn slice(&mut self, _addr: u32, _len: usize) -> Option<&[u8]> {
        None
    }

    /// the same bytes to write in place, where slice finds them.
    fn slice_mut(&mut self, _addr: u32, _len: usize) -> Option<&mut [u8]> {
        None
    }

    fn read_u32(&mut self, addr: u32) -> u32 {
        let mut buf = [0u8; 4];
        self.read(addr, &mut buf);
        u32::from_le_bytes(buf)
    }

    /// translates a GPU physical address to whatever address read/write
    /// expect. GX commands and registers all carry physical addresses.
    fn translate(&self, paddr: u32) -> u32 {
        paddr
    }

    /// the host GPU drew over a range that memory gets only when something
    /// asks for it, so the CPU's reads there have to ask first.
    fn guard(&mut self, _addr: u32, _len: u32) {}

    /// the same for the CPU's writes there.
    fn guard_writes(&mut self, _addr: u32, _len: u32) {}
}

/// what one LCD controller holds.
#[derive(Debug, Clone, Copy, Default)]
pub struct FramebufferConfig {
    pub address_a_left: u32,
    pub address_a_right: u32,
    pub address_b_left: u32,
    pub address_b_right: u32,
    pub stride: u32,
    pub format: u32,
    /// 0 selects the first pair of addresses, 1 the second.
    pub active: u32,
}

impl FramebufferConfig {
    /// address of the buffer currently being scanned out.
    pub fn address_left(&self) -> u32 {
        if self.active == 0 {
            self.address_a_left
        } else {
            self.address_b_left
        }
    }

    pub fn address_right(&self) -> u32 {
        if self.active == 0 {
            self.address_a_right
        } else {
            self.address_b_right
        }
    }

    /// the low three bits of the format register pick the color format, the
    /// rest configures the controller.
    pub fn color_format(&self) -> ColorFormat {
        ColorFormat::from_raw(self.format & 7)
    }

    /// true when the framebuffer is interleaved for stereoscopic output.
    pub fn is_stereo(&self) -> bool {
        let right = self.address_right();
        right != 0 && right != self.address_left()
    }
}

/// vertices a title sends one attribute at a time, by selecting fixed attribute
/// 15 and writing to the fixed attribute data registers.
#[derive(Default)]
struct Immediate {
    /// the vertex being assembled.
    attributes: [shader::Vec4; 16],
    next_attribute: usize,
    /// finished vertices, drawn together before the next change of state.
    vertices: Vec<[shader::Vec4; 16]>,
}

/// a screen's picture, upright RGBA, and the scale it is at.
pub type Picture = (std::sync::Arc<Vec<u8>>, u32);

/// the picture a display transfer left for a screen, drawn scaled by the
/// host's GPU, to be shown once the GPU finishes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenRef {
    /// the buffer the transfer drew, its address and size.
    pub(crate) addr: u32,
    pub(crate) size: (u32, u32),
    pub(crate) format: ColorFormat,
    pub(crate) batch: u64,
    /// the rows of it the screen shows, the first and how many, and how
    /// many pixels of each row, a buffer can have longer rows than the
    /// screen and start the screen a few rows in.
    pub(crate) rows: (u32, u32),
    pub(crate) columns: u32,
}

pub struct Gpu {
    /// external registers, 0x1EF00000 upwards, indexed by word.
    pub external: Box<[u32]>,
    /// PICA internal registers, written by command lists.
    pub internal: Box<[u32]>,
    /// the vertex shader unit, its program, uniforms and upload cursors.
    pub vertex_shader: shader::ShaderUnit,
    /// the unit that runs geometry shaders, when the pipeline uses one.
    pub geometry_shader: shader::ShaderUnit,
    /// values for attributes no array feeds, set through the fixed
    /// attribute registers.
    fixed_attributes: [shader::Vec4; 16],
    fixed_attribute_staging: [u32; 3],
    fixed_attribute_words: usize,
    /// vertices sent through the same registers in immediate mode.
    immediate: Immediate,
    /// where a command list asked execution to continue, as (physical
    /// address, size in bytes), taken once the current command finishes.
    pending_jump: Option<(u32, u32)>,

    /// one config per screen, index 0 is the top screen, 1 the bottom.
    pub framebuffers: [FramebufferConfig; 2],

    /// counters for the diagnostics overlay.
    pub command_lists: u64,
    pub draw_calls: u64,
    pub fills: u64,
    pub transfers: u64,
    /// vertices actually rasterized, as opposed to draw calls issued, a
    /// draw call with a degenerate vertex count still counts as a call.
    pub vertices_drawn: u64,
    /// host time spent running command lists and display transfers.
    pub busy: std::time::Duration,
    /// decoded textures and the lighting tables, kept across draws.
    resources: raster::Resources,
}

/// number of external register words we track (0x1EF00000..0x1EF04000).
const EXTERNAL_REGISTER_WORDS: usize = 0x1000;
/// PICA internal register file size.
const INTERNAL_REGISTER_WORDS: usize = 0x300;

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpu {
    pub fn new() -> Gpu {
        Gpu {
            external: vec![0; EXTERNAL_REGISTER_WORDS].into_boxed_slice(),
            internal: vec![0; INTERNAL_REGISTER_WORDS].into_boxed_slice(),
            vertex_shader: shader::ShaderUnit::new(),
            geometry_shader: shader::ShaderUnit::new(),
            fixed_attributes: [[0.0, 0.0, 0.0, 1.0]; 16],
            fixed_attribute_staging: [0; 3],
            fixed_attribute_words: 0,
            immediate: Immediate::default(),
            pending_jump: None,
            framebuffers: [FramebufferConfig::default(); 2],
            command_lists: 0,
            draw_calls: 0,
            fills: 0,
            transfers: 0,
            vertices_drawn: 0,
            busy: std::time::Duration::ZERO,
            resources: raster::Resources::default(),
        }
    }

    // -- external registers -------------------------------------------------

    /// offset is the byte offset GSP uses, where zero means 0x1EB00000.
    pub fn write_external(&mut self, gsp_offset: u32, value: u32) {
        let Some(index) = external_index(gsp_offset) else {
            return;
        };
        if index >= self.external.len() {
            return;
        }
        self.external[index] = value;
        let byte_offset = index * 4;
        if log::log_enabled!(log::Level::Trace)
            && (LCD_TOP_BASE..LCD_BOTTOM_BASE + 0x100).contains(&byte_offset)
        {
            log::trace!("GPU reg 0x1EF00{byte_offset:03X} = 0x{value:08X}");
        }
        self.on_external_write(byte_offset, value);
    }

    pub fn read_external(&self, gsp_offset: u32) -> u32 {
        external_index(gsp_offset)
            .and_then(|index| self.external.get(index).copied())
            .unwrap_or(0)
    }

    /// mirrors the LCD registers into our framebuffer state.
    fn on_external_write(&mut self, byte_offset: usize, value: u32) {
        for (screen, base) in [(0usize, LCD_TOP_BASE), (1, LCD_BOTTOM_BASE)] {
            let relative = byte_offset.wrapping_sub(base);
            let config = &mut self.framebuffers[screen];
            match relative {
                LCD_FB_A_LEFT => config.address_a_left = value,
                LCD_FB_A_RIGHT => config.address_a_right = value,
                LCD_FB_B_LEFT => config.address_b_left = value,
                LCD_FB_B_RIGHT => config.address_b_right = value,
                LCD_FB_FORMAT => config.format = value,
                LCD_FB_STRIDE => config.stride = value,
                LCD_FB_SELECT => config.active = value & 1,
                _ => {}
            }
        }
    }

    /// gsp::SetBufferSwap.
    pub fn set_framebuffer(
        &mut self,
        screen: u32,
        active: u32,
        left: u32,
        right: u32,
        stride: u32,
        format: u32,
    ) {
        let base = match screen {
            0 => LCD_TOP_BASE,
            1 => LCD_BOTTOM_BASE,
            _ => return,
        };
        let (left_offset, right_offset) = if active == 0 {
            (LCD_FB_A_LEFT, LCD_FB_A_RIGHT)
        } else {
            (LCD_FB_B_LEFT, LCD_FB_B_RIGHT)
        };
        for (offset, value) in [
            (left_offset, left),
            (right_offset, right),
            (LCD_FB_STRIDE, stride),
            (LCD_FB_FORMAT, format),
            (LCD_FB_SELECT, active),
        ] {
            let index = (base + offset) / 4;
            if index < self.external.len() {
                self.external[index] = value;
            }
            self.on_external_write(base + offset, value);
        }
    }

    // -- fill and transfer engines -----------------------------------------

    /// MemoryFill, writes a repeating pattern over a physical range.
    pub fn memory_fill<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        start: u32,
        end: u32,
        value: u32,
        width: u32,
    ) {
        if end <= start {
            return;
        }
        let length = (end - start) as usize;
        let address = memory.translate(start);
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            if let Err(error) = hardware.before_fill(memory, address, length as u32, if matches!(width, 2 | 3) { width } else { 4 }) {
                log::error!("the GPU could not write back a buffer, {error}");
            }
        }

        let bytes = value.to_le_bytes();
        let pattern = &bytes[..if matches!(width, 2 | 3) { width as usize } else { 4 }];
        log::debug!(
            "memory fill: 0x{start:08X}..0x{end:08X} with 0x{value:08X} ({width}-byte pattern)"
        );
        // in place where memory is in one piece, a frame's fills are
        // megabytes, else over the range in the last fill's buffer
        let mut buffer = std::mem::take(&mut self.resources.fill);
        match memory.slice_mut(address, length) {
            Some(range) => pattern::fill(range, pattern),
            None => {
                buffer.resize(length, 0);
                repeat(pattern, &mut buffer);
                memory.write(address, &buffer);
            }
        }
        self.fills += 1;
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            let filled = match memory.slice(address, length) {
                Some(range) => range,
                None => &buffer[..],
            };
            if let Err(error) = hardware.filled(address, filled) {
                log::error!("the GPU could not clear a buffer, {error}");
            }
        }
        self.resources.fill = buffer;
    }

    /// DisplayTransfer, copies a rectangle between buffers, converting format
    /// and tiling on the way.
    pub fn display_transfer<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        input_paddr: u32,
        output_paddr: u32,
        input_dimensions: u32,
        output_dimensions: u32,
        flags: u32,
    ) {
        let input_width = input_dimensions & 0xFFFF;
        let input_height = input_dimensions >> 16;

        // flag layout, from the transfer engine's register,
        //   bit 0      flip the input vertically
        //   bit 1      linear to tiled, tiled to linear without it
        //   bit 2      crop the input's lines to the output's width
        //   bit 3      raw copy, no format conversion
        //   bit 5      keep the layout, tiled or linear on both sides
        //   bits 8-10  input color format
        //   bits 12-14 output color format
        //   bits 24-25 downscale
        let flip_vertically = flags & 1 != 0;
        let input_linear = flags & (1 << 1) != 0;
        let output_tiled = input_linear != (flags & (1 << 5) != 0);
        let input_format = ColorFormat::from_raw((flags >> 8) & 7);
        let output_format = ColorFormat::from_raw((flags >> 12) & 7);
        let downscale = (flags >> 24) & 3;

        let input_bpp = input_format.bytes_per_pixel();
        let output_bpp = output_format.bytes_per_pixel();

        let (scale_x, scale_y) = match downscale {
            1 => (2u32, 1u32),
            2 => (2, 2),
            _ => (1, 1),
        };
        // the output's size counts input pixels, a downscale shrinks it
        let output_width = (output_dimensions & 0xFFFF) / scale_x;
        let output_height = (output_dimensions >> 16) / scale_y;

        let copy_width = output_width.min(input_width / scale_x);
        let copy_height = output_height.min(input_height / scale_y);

        // flipped and cropped, the console starts the output as many bytes
        // later as the lines cropped off take, all but the last one. titles
        // count on it, Tales of the Abyss hands it its framebuffers that much
        // earlier
        let crop = flags & (1 << 2) != 0;
        let skew = if flip_vertically && crop {
            input_width.saturating_sub(output_dimensions & 0xFFFF) * output_height.saturating_sub(1) * output_bpp as u32
        } else {
            0
        };
        let output_paddr = output_paddr.wrapping_add(skew);
        let (input_base, output_base) = (memory.translate(input_paddr), memory.translate(output_paddr));

        log::debug!(
            "display transfer: 0x{input_paddr:08X} {input_width}x{input_height} \
             {input_format:?} {} -> 0x{output_paddr:08X} {output_width}x{output_height} \
             {output_format:?} {} (flags 0x{flags:08X})",
            if input_linear { "linear" } else { "tiled" },
            if output_tiled { "tiled" } else { "linear" },
        );

        // what the host GPU drew it transfers itself, keeping the output
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            let transfer = raster::hardware::Transfer {
                input: input_base,
                output: output_base,
                input_width,
                output_width,
                output_height,
                copy: (copy_width, copy_height),
                scale: (scale_x, scale_y),
                flip: flip_vertically,
                input_linear,
                output_tiled,
                input_format,
                output_format,
            };
            match hardware.display_transfer(memory, &transfer) {
                Ok(true) => {
                    self.transfers += 1;
                    return;
                }
                Ok(false) => {}
                Err(error) => log::error!("the GPU could not transfer a buffer, {error}"),
            }
        }
        // four bytes a pixel covers every format either side uses
        self.sync_memory(memory, input_base, input_width * input_height * 4);
        self.sync_memory(memory, output_base, output_width * output_height * 4);
        if copy_width == 0 || copy_height == 0 {
            return;
        }

        let mut input = vec![0u8; (input_width * input_height) as usize * input_bpp];
        memory.read(input_base, &mut input);
        let mut output = vec![0u8; (output_width * output_height) as usize * output_bpp];
        // preserve whatever was already there outside the copied rectangle.
        memory.read(output_base, &mut output);

        // halving a tiled RGBA8 buffer into another, which titles do to a
        // frame every frame for their effects, goes a tile at a time
        let whole_tiles = [copy_width, copy_height, input_width, output_width].iter().all(|n| n % 8 == 0);
        if input_format == ColorFormat::Rgba8
            && output_format == ColorFormat::Rgba8
            && !input_linear
            && output_tiled
            && !flip_vertically
            && (scale_x, scale_y) == (2, 2)
            && whole_tiles
            && copy_width * 2 <= input_width
            && copy_height * 2 <= input_height
        {
            halve_tiles(&input, input_width, &mut output, output_width, (copy_width, copy_height));
            memory.write(output_base, &output);
            self.transfers += 1;
            return;
        }

        let trace_pixels = std::env::var("ZAKURO_TRACE_PIXELS").is_ok();
        let mut distinct = std::collections::HashSet::new();

        // the input's rows are read from the first, a flip writes them from
        // the output's last row up
        for y in 0..copy_height {
            let out_y = if flip_vertically { output_height - 1 - y } else { y };
            for x in 0..copy_width {
                let dst = if output_tiled {
                    morton(x, out_y, output_width, output_bpp)
                } else {
                    (out_y * output_width + x) as usize * output_bpp
                };
                if dst + output_bpp > output.len() {
                    continue;
                }

                // a downscale averages each 2x1 or 2x2 block
                let mut sum = [0u32; 4];
                let mut count = 0u32;
                for dy in 0..scale_y {
                    for dx in 0..scale_x {
                        let src_x = x * scale_x + dx;
                        let src_y = y * scale_y + dy;
                        let src = if input_linear {
                            (src_y * input_width + src_x) as usize * input_bpp
                        } else {
                            morton(src_x, src_y, input_width, input_bpp)
                        };
                        if src + input_bpp > input.len() {
                            continue;
                        }
                        let sample = input_format.decode(&input[src..src + input_bpp]);
                        for (total, value) in sum.iter_mut().zip(sample) {
                            *total += value as u32;
                        }
                        count += 1;
                    }
                }
                if count == 0 {
                    continue;
                }
                let pixel = sum.map(|total| ((total + count / 2) / count) as u8);
                if trace_pixels {
                    distinct.insert(pixel);
                }
                output_format.encode(pixel, &mut output[dst..dst + output_bpp]);
            }
        }
        if trace_pixels {
            log::debug!(
                "display transfer input 0x{input_paddr:08X}: {} distinct pixel(s), sample {:?}",
                distinct.len(),
                distinct.iter().take(5).collect::<Vec<_>>()
            );
        }

        memory.write(output_base, &output);
        self.transfers += 1;
    }

    /// TextureCopy, a raw byte copy with a stride, no format conversion.
    pub fn texture_copy<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        input_paddr: u32,
        output_paddr: u32,
        size: u32,
        input_gap: u32,
        output_gap: u32,
    ) {
        let input_width = (input_gap & 0xFFFF) * 16;
        let input_skip = (input_gap >> 16) * 16;
        // every line and the gaps between them
        let span = |width: u32, skip: u32| if width == 0 { size } else { size + size.div_ceil(width) * skip };
        let (input, output) = (memory.translate(input_paddr), memory.translate(output_paddr));
        self.sync_memory(memory, input, span(input_width, input_skip));
        let output_width = (output_gap & 0xFFFF) * 16;
        let output_skip = (output_gap >> 16) * 16;
        self.sync_memory(memory, output, span(output_width, output_skip));

        log::debug!(
            "texture copy: 0x{input_paddr:08X} -> 0x{output_paddr:08X} size 0x{size:X}"
        );

        let (mut src, mut dst) = (input, output);

        // a zero width means one contiguous run.
        if input_width == 0 || output_width == 0 {
            let mut buffer = vec![0u8; size as usize];
            memory.read(src, &mut buffer);
            memory.write(dst, &buffer);
            self.transfers += 1;
            return;
        }

        let mut remaining = size;
        let chunk = input_width.min(output_width);
        let mut buffer = vec![0u8; chunk as usize];
        while remaining >= chunk && chunk > 0 {
            memory.read(src, &mut buffer);
            memory.write(dst, &buffer);
            src += input_width + input_skip;
            dst += output_width + output_skip;
            remaining -= chunk;
        }
        self.transfers += 1;
    }

    // -- command lists ------------------------------------------------------

    /// walks a command list, applying its register writes and dispatching any
    /// draw it triggers to the software rasterizer.
    pub fn process_command_list<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        paddr: u32,
        size: u32,
    ) {
        let start = std::time::Instant::now();
        self.resources.textures.begin_list();
        self.run_command_list(memory, renderer, paddr, size);
        self.busy += start.elapsed();
    }

    /// draws on the host's GPU from now on, rather than in software, and
    /// says which GPU that is.
    /// scale is how many times the console's resolution it draws at, for
    /// sharper pictures. on the device of a Vulkan presenter, when it gives
    /// one, the screens are shown straight from the images drawn.
    #[cfg(feature = "vulkan")]
    pub fn enable_hardware_renderer(&mut self, scale: u32, device: Option<std::sync::Arc<SharedDevice>>) -> Result<String, String> {
        let mut hardware = match device {
            Some(device) => raster::hardware::Hardware::with_device(device, true)?,
            None => raster::hardware::Hardware::new()?,
        };
        let scale = hardware.set_scale(scale);
        let shaded = if hardware.shades() { "" } else { ", vertices shaded on the CPU," };
        let direct = if hardware.direct() { ", shown straight from the GPU," } else { "" };
        let name = format!("{} at {scale}x{shaded}{direct}", hardware.name());
        self.resources.hardware = Some(hardware);
        Ok(name)
    }

    /// draws the texture pack's pictures in place of the textures they
    /// replace, or no pack's. only the host's GPU draws them, so a pack
    /// waits for enable_hardware_renderer, and says whether it is drawn.
    pub fn set_texture_pack(&mut self, pack: Option<std::sync::Arc<pack::Pack>>) -> bool {
        #[cfg(feature = "vulkan")]
        let pack = pack.filter(|_| self.resources.hardware.is_some());
        #[cfg(not(feature = "vulkan"))]
        let pack = pack.filter(|_| false);
        let drawn = pack.is_some();
        self.resources.textures.set_pack(pack);
        drawn
    }

    /// whether the screens the host's GPU draws are shown straight from it,
    /// by a presenter sharing its device, see screen_image.
    pub fn shows_directly(&self) -> bool {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_ref() {
            return hardware.direct();
        }
        false
    }

    /// where a screen's picture is on the GPU, for a presenter sharing the
    /// device to draw it straight from, the batch that drew it submitted.
    pub fn screen_image(&mut self, screen: ScreenRef) -> Option<GpuScreen> {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            match hardware.screen_image(screen) {
                Ok(image) => return image,
                Err(error) => log::error!("the GPU could not show a screen, {error}"),
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = screen;
        None
    }

    /// how many times the console's resolution the host's GPU draws at, 1
    /// when drawing in software.
    pub fn scale(&self) -> u32 {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_ref() {
            return hardware.scale();
        }
        1
    }

    /// the newest picture the host's GPU drew for a screen's buffer, at the
    /// scale it draws at, while guest memory still holds the same picture.
    /// size is the pixels of a row the screen shows and its rows, stride the
    /// pixels a row takes in memory.
    pub fn scaled_screen(&mut self, addr: u32, size: (u32, u32), stride: u32, format: ColorFormat, guest: &[u8]) -> Option<ScreenRef> {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            return hardware.screen(addr, size, stride, format, guest);
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = (addr, size, stride, format, guest);
        None
    }

    /// a screen's picture upright, RGBA the way the screen shows it, and the
    /// scale it is at, once the GPU finished it.
    pub fn scaled_picture(&mut self, screen: ScreenRef) -> Option<Picture> {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            match hardware.picture(screen) {
                Ok(picture) => return picture,
                Err(error) => log::error!("the GPU could not show a screen, {error}"),
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = screen;
        None
    }

    /// makes guest memory right over a range something other than a draw
    /// is about to read or write, the host GPU keeps what it draws until then.
    pub fn sync_memory<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            if let Err(error) = hardware.sync(memory, addr, len) {
                log::error!("the GPU could not write back a buffer, {error}");
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = (memory, addr, len);
    }

    /// makes guest memory hold what the host GPU drew of the depth buffers
    /// over a range, for the CPU reading there. the CPU reads color buffers
    /// as memory holds them.
    pub fn sync_depth<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.resources.hardware.as_mut() {
            if let Err(error) = hardware.sync_depth(memory, addr, len) {
                log::error!("the GPU could not write back a buffer, {error}");
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = (memory, addr, len);
    }

    fn run_command_list<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        paddr: u32,
        size: u32,
    ) {
        // into the last list's buffer, a list a few times a frame, large
        // when draws upload their uniforms, would have a new block each time
        let mut words = std::mem::take(&mut self.resources.words);
        read_command_buffer(memory, paddr, size, &mut words);
        // buffers jump into each other, and a buffer jumping to itself
        // would never end, real lists stay far below this.
        let mut jumps = 0u32;
        self.pending_jump = None;

        let mut index = 0usize;
        while index + 1 < words.len() {
            let data = words[index];
            let header = words[index + 1];
            index += 2;

            let register = (header & 0xFFFF) as usize;
            let mask = (header >> 16) & 0xF;
            let extra = ((header >> 20) & 0xFF) as usize;
            let consecutive = header & 0x8000_0000 != 0;

            // no real PICA register lives anywhere near this index (the whole
            // internal file is 0x300 words), seeing one would mean a desync in
            // the parsing above.
            if register >= self.internal.len() {
                log::debug!(
                    "command list stops at word {}/{}: register 0x{register:04X} is out of range",
                    index - 2,
                    words.len()
                );
                break;
            }

            // a list cut short ends with the words it has
            let burst = &words[index..(index + extra).min(words.len())];
            self.run_command(memory, renderer, register, mask, consecutive, data, burst);
            index += burst.len();

            // each command is padded so its total length (the base pair plus
            // extra words) is a multiple of two, an odd extra needs one
            // more word to reach that, an even one is already there.
            if !extra.is_multiple_of(2) {
                index += 1;
            }
            index = index.min(words.len());

            // a jump replaces the list being executed.
            if let Some((address, size)) = self.pending_jump.take() {
                jumps += 1;
                if jumps > 0x10000 {
                    log::warn!("command list jumps more than {jumps} times; stopping");
                    break;
                }
                read_command_buffer(memory, address, size, &mut words);
                index = 0;
            }
        }
        self.resources.words = words;

        self.flush_immediate(memory);
        self.command_lists += 1;
    }

    /// one command, its first word and the burst after it, all to one
    /// register or on through the ones after it. what titles send most,
    /// runs of state and runs for one of the ports, goes in one go, anything
    /// else a write at a time. lists are mostly these, and a write at a time
    /// took longer than the rest of the list.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn run_command<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        register: usize,
        mask: u32,
        consecutive: bool,
        first: u32,
        burst: &[u32],
    ) {
        // the registers from the first on that only hold state take their
        // words right here
        let plain = PLAIN_UNTIL[register] as usize - register;
        if plain > 0 {
            self.flush_immediate(memory);
            let bits = BYTE_MASKS[mask as usize];
            if !consecutive {
                // each word writes the same bytes, the last one's stay
                let value = burst.last().copied().unwrap_or(first);
                let slot = &mut self.internal[register];
                *slot = (*slot & !bits) | (value & bits);
                return;
            }
            let count = plain.min(burst.len() + 1);
            let slots = &mut self.internal[register..register + count];
            slots[0] = (slots[0] & !bits) | (first & bits);
            for (slot, &value) in slots[1..].iter_mut().zip(burst) {
                *slot = (*slot & !bits) | (value & bits);
            }
            // whatever the burst runs on into a write at a time
            for (i, &value) in burst.iter().enumerate().skip(count - 1) {
                self.write_internal(memory, renderer, register + 1 + i, value, mask);
            }
            return;
        }
        if mask == 0xF && self.write_port(memory, renderer, register, consecutive, first, burst) {
            return;
        }
        self.write_internal(memory, renderer, register, first, mask);
        for (i, &value) in burst.iter().enumerate() {
            let target = if consecutive { register + 1 + i } else { register };
            self.write_internal(memory, renderer, target, value, mask);
        }
    }

    /// a command writing whole words that all go to one port, a shader
    /// unit's float uniforms, program or operand descriptors, or one of the
    /// tables, taken as one run. false when it is not one. a float uniform
    /// upload can start at the index register and run on into the data
    /// registers, and only the last of several words to the index counts.
    fn write_port<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        register: usize,
        consecutive: bool,
        first: u32,
        burst: &[u32],
    ) -> bool {
        let last = if consecutive { register + burst.len() } else { register };
        let words = || std::iter::once(&first).chain(burst);
        match register {
            REG_VS_BLOCK..=REG_VS_BLOCK_END | REG_GS_BLOCK..=REG_GS_BLOCK_END => {
                let vertex = register >= REG_VS_BLOCK;
                let block = if vertex { REG_VS_BLOCK } else { REG_GS_BLOCK };
                let (offset, end) = (register - block, last - block);
                match offset {
                    SHADER_UNIFORM_INDEX if !consecutive => {
                        self.flush_immediate(memory);
                        let (unit, _) = self.units(vertex);
                        unit.set_float_uniform_index(burst.last().copied().unwrap_or(first));
                    }
                    SHADER_UNIFORM_INDEX if end <= SHADER_UNIFORM_DATA_END => {
                        self.flush_immediate(memory);
                        let (unit, _) = self.units(vertex);
                        unit.set_float_uniform_index(first);
                        unit.upload_float_uniforms(burst);
                    }
                    SHADER_UNIFORM_DATA..=SHADER_UNIFORM_DATA_END if end <= SHADER_UNIFORM_DATA_END => {
                        self.flush_immediate(memory);
                        let (unit, _) = self.units(vertex);
                        unit.upload_float_uniform(first);
                        unit.upload_float_uniforms(burst);
                    }
                    SHADER_PROGRAM_DATA..=SHADER_PROGRAM_DATA_END if end <= SHADER_PROGRAM_DATA_END => {
                        self.flush_immediate(memory);
                        // as in write_internal, the geometry unit takes the
                        // vertex shader's program and descriptors too
                        let mirror = vertex && self.internal[REG_VS_COM_MODE] & 1 == 0;
                        let (unit, other) = self.units(vertex);
                        for &word in words() {
                            unit.upload_program(word);
                            if mirror {
                                other.upload_program(word);
                            }
                            if vertex {
                                renderer.upload_shader_code(word);
                            }
                        }
                    }
                    SHADER_DESCRIPTOR_DATA..=SHADER_DESCRIPTOR_DATA_END if end <= SHADER_DESCRIPTOR_DATA_END => {
                        self.flush_immediate(memory);
                        let mirror = vertex && self.internal[REG_VS_COM_MODE] & 1 == 0;
                        let (unit, other) = self.units(vertex);
                        for &word in words() {
                            unit.upload_descriptor(word);
                            if mirror {
                                other.upload_descriptor(word);
                            }
                            if vertex {
                                renderer.upload_shader_operand_descriptor(word);
                            }
                        }
                    }
                    _ => return false,
                }
            }
            lighting::REG_TABLE_DATA..=lighting::REG_TABLE_DATA_END if last <= lighting::REG_TABLE_DATA_END => {
                self.flush_immediate(memory);
                for &word in words() {
                    self.resources.light_tables.write(&mut self.internal, word);
                }
            }
            proctex::REG_TABLE_DATA..=proctex::REG_TABLE_DATA_END if last <= proctex::REG_TABLE_DATA_END => {
                self.flush_immediate(memory);
                for &word in words() {
                    self.resources.proctex_tables.write(&mut self.internal, word);
                }
            }
            fog::REG_TABLE_DATA..=fog::REG_TABLE_DATA_END if last <= fog::REG_TABLE_DATA_END => {
                self.flush_immediate(memory);
                for &word in words() {
                    self.resources.fog_table.write(&mut self.internal, word);
                }
            }
            _ => return false,
        }
        // the registers keep the last word each took, nothing the ports do
        // reads them
        if consecutive {
            self.internal[register] = first;
            self.internal[register + 1..=last].copy_from_slice(burst);
        } else {
            self.internal[register] = burst.last().copied().unwrap_or(first);
        }
        true
    }

    /// the vertex shader unit and the geometry one, or the other way round.
    fn units(&mut self, vertex: bool) -> (&mut shader::ShaderUnit, &mut shader::ShaderUnit) {
        if vertex {
            (&mut self.vertex_shader, &mut self.geometry_shader)
        } else {
            (&mut self.geometry_shader, &mut self.vertex_shader)
        }
    }

    /// decodes the programs a draw runs. the geometry unit gets the vertex
    /// shader's uploads too, and decoding hashes a whole program, so it is
    /// left alone while the geometry stage is off.
    fn prepare_shaders(&mut self) {
        self.vertex_shader.prepare();
        if self.internal[REG_GEOSTAGE_CONFIG] & 0x3 == 2 {
            self.geometry_shader.prepare();
        }
    }

    /// draws any immediate-mode vertices still waiting. every write but the
    /// attribute data asks, and there seldom are any.
    #[inline]
    fn flush_immediate<M: GpuMemory>(&mut self, memory: &mut M) {
        if !self.immediate.vertices.is_empty() {
            self.draw_immediate(memory);
        }
    }

    /// draws what the registers describe, the vertices of the arrays in
    /// order or the ones the index buffer picks. kept out of write_internal,
    /// where every write paid for the stack a draw needs.
    #[inline(never)]
    fn draw<M: GpuMemory>(&mut self, memory: &mut M, renderer: &mut dyn Renderer, indexed: bool) {
        self.draw_calls += 1;
        renderer.draw(DrawCall {
            indexed,
            registers: &self.internal,
        });
        self.prepare_shaders();
        let vertices = raster::draw(
            &self.internal,
            &self.vertex_shader,
            &self.geometry_shader,
            &self.fixed_attributes,
            memory,
            &mut self.resources,
            indexed,
        );
        self.vertices_drawn += vertices as u64;
    }

    /// draws the immediate-mode vertices waiting, see flush_immediate.
    #[inline(never)]
    fn draw_immediate<M: GpuMemory>(&mut self, memory: &mut M) {
        let vertices = std::mem::take(&mut self.immediate.vertices);
        self.draw_calls += 1;
        self.vertices_drawn += vertices.len() as u64;
        self.prepare_shaders();
        raster::draw_immediate(
            &self.internal,
            &self.vertex_shader,
            &self.geometry_shader,
            memory,
            &mut self.resources,
            &vertices,
        );
    }

    fn write_internal<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        register: usize,
        value: u32,
        mask: u32,
    ) {
        if register >= self.internal.len() {
            return;
        }

        // immediate-mode vertices are drawn with the state they were sent
        // under, so anything else touching a register draws them first.
        if !(REG_FIXED_ATTRIBUTE_DATA..=REG_FIXED_ATTRIBUTE_DATA_END).contains(&register) {
            self.flush_immediate(memory);
        }

        // the mask selects which bytes of the register the write touches, a set
        // bit writes that byte.
        let write_mask = BYTE_MASKS[mask as usize & 0xF];
        let old = self.internal[register];
        let new = (old & !write_mask) | (value & write_mask);
        self.internal[register] = new;

        // a register given an effect here has to leave plain() too, or the
        // commands run_command takes in one go skip it
        match register {
            REG_DRAW_ARRAYS | REG_DRAW_ELEMENTS => self.draw(memory, renderer, register == REG_DRAW_ELEMENTS),

            REG_CMDBUF_JUMP0 | REG_CMDBUF_JUMP1 => {
                // every 3D model lives behind one of these jumps. spent way too long
                // wondering why the professor wasn't there, fml
                let channel = register - REG_CMDBUF_JUMP0;
                let size = (self.internal[REG_CMDBUF_SIZE0 + channel] & 0x1F_FFFF) * 8;
                let address = (self.internal[REG_CMDBUF_ADDR0 + channel] & 0x1FFF_FFFF) * 8;
                self.pending_jump = Some((address, size));
            }

            REG_FIXED_ATTRIBUTE_INDEX => {
                self.fixed_attribute_words = 0;
                self.immediate.next_attribute = 0;
            }
            REG_FIXED_ATTRIBUTE_DATA..=REG_FIXED_ATTRIBUTE_DATA_END => {
                self.fixed_attribute_staging[self.fixed_attribute_words] = new;
                self.fixed_attribute_words += 1;
                if self.fixed_attribute_words == 3 {
                    self.fixed_attribute_words = 0;
                    // packed like float uniforms, four 24-bit floats in
                    // three words, w first.
                    let [a, b, c] = self.fixed_attribute_staging;
                    let attribute = [
                        shader::isa::decode_float24(c & 0x00FF_FFFF),
                        shader::isa::decode_float24(((b & 0xFFFF) << 8) | (c >> 24)),
                        shader::isa::decode_float24(((a & 0xFF) << 16) | (b >> 16)),
                        shader::isa::decode_float24(a >> 8),
                    ];
                    let index = (self.internal[REG_FIXED_ATTRIBUTE_INDEX] & 0xF) as usize;
                    if index < 15 {
                        self.fixed_attributes[index] = attribute;
                        // the index moves on so consecutive attributes can
                        // be set in one burst.
                        self.internal[REG_FIXED_ATTRIBUTE_INDEX] =
                            (self.internal[REG_FIXED_ATTRIBUTE_INDEX] & !0xF) | (index as u32 + 1);
                    } else {
                        // immediate mode, attributes arrive in order, and
                        // the last one completes a vertex.
                        let immediate = &mut self.immediate;
                        immediate.attributes[immediate.next_attribute] = attribute;
                        let last = (self.internal[REG_VS_ATTRIBUTE_COUNT] & 0xF) as usize;
                        if immediate.next_attribute < last {
                            immediate.next_attribute += 1;
                        } else {
                            immediate.next_attribute = 0;
                            immediate.vertices.push(immediate.attributes);
                        }
                    }
                }
            }

            lighting::REG_TABLE_DATA..=lighting::REG_TABLE_DATA_END => {
                self.resources.light_tables.write(&mut self.internal, new);
            }
            proctex::REG_TABLE_DATA..=proctex::REG_TABLE_DATA_END => {
                self.resources.proctex_tables.write(&mut self.internal, new);
            }
            fog::REG_TABLE_DATA..=fog::REG_TABLE_DATA_END => {
                self.resources.fog_table.write(&mut self.internal, new);
            }
            REG_GS_BLOCK..=REG_GS_BLOCK_END => {
                configure_shader(&mut self.geometry_shader, register - REG_GS_BLOCK, new);
            }
            REG_VS_BLOCK..=REG_VS_BLOCK_END => {
                let offset = register - REG_VS_BLOCK;
                configure_shader(&mut self.vertex_shader, offset, new);
                // the geometry shader unit takes the vertex shader's program
                // and descriptors as well, unless the title configures it on
                // its own.
                let program = matches!(
                    offset,
                    SHADER_PROGRAM_INDEX..=SHADER_PROGRAM_DATA_END
                        | SHADER_DESCRIPTOR_INDEX..=SHADER_DESCRIPTOR_DATA_END
                );
                if program && self.internal[REG_VS_COM_MODE] & 1 == 0 {
                    configure_shader(&mut self.geometry_shader, offset, new);
                }
                match offset {
                    SHADER_PROGRAM_DATA..=SHADER_PROGRAM_DATA_END => renderer.upload_shader_code(new),
                    SHADER_DESCRIPTOR_DATA..=SHADER_DESCRIPTOR_DATA_END => {
                        renderer.upload_shader_operand_descriptor(new)
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

/// pattern over bytes, from the start again each time it ends, the copies
/// doubling.
fn repeat(pattern: &[u8], bytes: &mut [u8]) {
    let first = pattern.len().min(bytes.len());
    bytes[..first].copy_from_slice(&pattern[..first]);
    let mut done = first;
    while done < bytes.len() {
        let more = done.min(bytes.len() - done);
        bytes.copy_within(..more, done);
        done += more;
    }
}

/// reads a command buffer as words, into words.
fn read_command_buffer<M: GpuMemory>(memory: &mut M, paddr: u32, size: u32, words: &mut Vec<u32>) {
    let addr = memory.translate(paddr);
    words.clear();
    match memory.slice(addr, size as usize) {
        Some(bytes) => words.extend(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c))),
        None => {
            let mut bytes = vec![0u8; size as usize];
            memory.read(addr, &mut bytes);
            words.extend(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
        }
    }
}

const REG_GS_BLOCK_END: usize = REG_GS_BLOCK + SHADER_BLOCK_SIZE - 1;
const REG_VS_BLOCK_END: usize = REG_VS_BLOCK + SHADER_BLOCK_SIZE - 1;

/// the bits of a register each of the sixteen byte masks of a command
/// writes, a set bit of the mask writing that byte.
const BYTE_MASKS: [u32; 16] = {
    let mut masks = [0; 16];
    let mut mask = 0;
    while mask < 16 {
        let mut byte = 0;
        while byte < 4 {
            if mask & (1 << byte) != 0 {
                masks[mask] |= 0xFF << (byte * 8);
            }
            byte += 1;
        }
        mask += 1;
    }
    masks
};

/// for each register, the first one from it on that a write does more to
/// than store the value, the writes of a command up to there only store.
static PLAIN_UNTIL: [u16; INTERNAL_REGISTER_WORDS] = {
    let mut until = [0; INTERNAL_REGISTER_WORDS];
    let mut end = INTERNAL_REGISTER_WORDS;
    let mut register = INTERNAL_REGISTER_WORDS;
    while register > 0 {
        register -= 1;
        if !plain(register) {
            end = register;
        }
        until[register] = end as u16;
    }
    until
};

/// whether a write to a register only stores the value, the ones that do
/// more are the arms of write_internal and configure_shader.
const fn plain(register: usize) -> bool {
    match register {
        REG_DRAW_ARRAYS | REG_DRAW_ELEMENTS | REG_CMDBUF_JUMP0 | REG_CMDBUF_JUMP1 => false,
        REG_FIXED_ATTRIBUTE_INDEX..=REG_FIXED_ATTRIBUTE_DATA_END => false,
        lighting::REG_TABLE_DATA..=lighting::REG_TABLE_DATA_END
        | proctex::REG_TABLE_DATA..=proctex::REG_TABLE_DATA_END
        | fog::REG_TABLE_DATA..=fog::REG_TABLE_DATA_END => false,
        REG_GS_BLOCK..=REG_GS_BLOCK_END => !configures_shader(register - REG_GS_BLOCK),
        REG_VS_BLOCK..=REG_VS_BLOCK_END => !configures_shader(register - REG_VS_BLOCK),
        _ => true,
    }
}

/// whether a write to a shader unit's register at an offset changes the
/// unit, the vertex unit's program and descriptor ones the geometry unit
/// too.
const fn configures_shader(offset: usize) -> bool {
    matches!(
        offset,
        SHADER_BOOL_UNIFORMS
            | SHADER_INT_UNIFORMS..=SHADER_INT_UNIFORMS_END
            | SHADER_ENTRY_POINT
            | SHADER_UNIFORM_INDEX..=SHADER_UNIFORM_DATA_END
            | SHADER_PROGRAM_INDEX..=SHADER_PROGRAM_DATA_END
            | SHADER_DESCRIPTOR_INDEX..=SHADER_DESCRIPTOR_DATA_END
    )
}

/// applies a write to one of a shader unit's configuration registers.
fn configure_shader(unit: &mut shader::ShaderUnit, offset: usize, value: u32) {
    match offset {
        SHADER_BOOL_UNIFORMS => unit.bool_uniforms = (value & 0xFFFF) as u16,
        SHADER_INT_UNIFORMS..=SHADER_INT_UNIFORMS_END => {
            unit.int_uniforms[offset - SHADER_INT_UNIFORMS] = value.to_le_bytes();
        }
        SHADER_ENTRY_POINT => unit.entry_point = value & (shader::PROGRAM_SIZE as u32 - 1),
        SHADER_UNIFORM_INDEX => unit.set_float_uniform_index(value),
        SHADER_UNIFORM_DATA..=SHADER_UNIFORM_DATA_END => unit.upload_float_uniform(value),
        SHADER_PROGRAM_INDEX => {
            unit.program_write_offset = value as usize & (shader::PROGRAM_SIZE - 1);
        }
        SHADER_PROGRAM_DATA..=SHADER_PROGRAM_DATA_END => unit.upload_program(value),
        SHADER_DESCRIPTOR_INDEX => {
            unit.descriptor_write_offset = value as usize & (shader::DESCRIPTOR_SIZE - 1);
        }
        SHADER_DESCRIPTOR_DATA..=SHADER_DESCRIPTOR_DATA_END => unit.upload_descriptor(value),
        _ => {}
    }
}

#[inline]
fn morton(x: u32, y: u32, width: u32, bytes_per_pixel: usize) -> usize {
    format::morton_offset(x, y, width, bytes_per_pixel as u32) as usize
}

/// the top left of a tiled RGBA8 buffer halved each way into another, size
/// pixels of the output, each the rounded average of four, as the transfer
/// does it a pixel at a time. the four follow one another in their tile, and
/// a tile of the input makes a quarter of one of the output.
fn halve_tiles(input: &[u8], input_width: u32, output: &mut [u8], output_width: u32, (width, height): (u32, u32)) {
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            let tile = morton(x, y, output_width, 4);
            for quarter in 0..4 {
                let (dx, dy) = (quarter & 1, quarter >> 1);
                let from = morton(x * 2 + dx * 8, y * 2 + dy * 8, input_width, 4);
                let to = tile + (dx as usize * 16 + dy as usize * 32) * 4;
                let pixels = input[from..from + 256].as_chunks::<16>().0;
                for (four, out) in pixels.iter().zip(output[to..to + 64].as_chunks_mut::<4>().0) {
                    for (channel, value) in out.iter_mut().enumerate() {
                        let sum: u32 = (0..4).map(|pixel| four[pixel * 4 + channel] as u32).sum();
                        *value = ((sum + 2) / 4) as u8;
                    }
                }
            }
        }
    }
}

/// converts a GSP register offset to an index into our external register file.
fn external_index(gsp_offset: u32) -> Option<usize> {
    // GSP offsets are relative to 0x1EB00000, but the GPU's registers start at
    // 0x1EF00000, so the first 0x400000 bytes are other hardware.
    let byte_offset = gsp_offset.checked_sub(0x0040_0000)?;
    Some(byte_offset as usize / 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// guest memory as one flat block starting at address zero.
    #[derive(Clone)]
    struct FlatMemory(Vec<u8>);

    impl GpuMemory for FlatMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            let start = addr as usize;
            out.copy_from_slice(&self.0[start..start + out.len()]);
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            let start = addr as usize;
            self.0[start..start + data.len()].copy_from_slice(data);
        }
    }

    /// the inverse of [shader::isa::decode_float24], for normal values.
    fn float24(value: f32) -> u32 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let sign = bits >> 31;
        let exponent = ((bits >> 23) & 0xFF) + 63 - 127;
        (sign << 23) | (exponent << 16) | ((bits >> 7) & 0xFFFF)
    }

    /// packs a vector the way the fixed attribute registers take it.
    fn pack(v: [f32; 4]) -> [u32; 3] {
        let [x, y, z, w] = v.map(float24);
        [(w << 8) | (z >> 16), ((z & 0xFFFF) << 16) | (y >> 8), ((y & 0xFF) << 24) | x]
    }

    const COLOR_BUFFER: u32 = 0x1000;

    /// A GPU set up to draw an 8x8 RGBA8 target with a shader that passes
    /// v0 through as the position and v1 as the color.
    fn gpu_for_immediate_draws(memory: &mut FlatMemory, renderer: &mut dyn Renderer) -> Gpu {
        let mut gpu = Gpu::new();
        let mov = 0x13 << 26;
        gpu.vertex_shader.program[..3]
            .copy_from_slice(&[mov, mov | (1 << 21) | (1 << 12), 0x22 << 26]);
        gpu.vertex_shader.descriptors[0] = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);

        let mut write = |register: usize, value: u32| {
            gpu.write_internal(memory, renderer, register, value, 0xF)
        };
        write(REG_SHADER_OUTPUT_TOTAL, 2);
        write(REG_SHADER_OUTPUT_MAP, 0x0302_0100); // o0 = position
        write(REG_SHADER_OUTPUT_MAP + 1, 0x0B0A_0908); // o1 = color
        write(REG_VS_INPUT_REGISTER_MAP_LOW, 0x10); // attribute n -> vn
        write(REG_VS_ATTRIBUTE_COUNT, 1); // two attributes per vertex
        write(REG_VIEWPORT_WIDTH, float24(4.0));
        write(REG_VIEWPORT_HEIGHT, float24(4.0));
        write(REG_COLOR_BUFFER_ADDRESS, COLOR_BUFFER >> 3);
        write(REG_FRAMEBUFFER_DIMENSIONS, 8 | (7 << 12));
        // color writes allowed, all four channels.
        write(REG_COLOR_BUFFER_WRITE, 0xF);
        write(REG_DEPTH_COLOR_MASK, 0xF << 8);
        write(REG_LOGIC_OP, 3);
        gpu
    }

    fn send_vertex(gpu: &mut Gpu, memory: &mut FlatMemory, renderer: &mut SoftwareRenderer, attributes: &[[f32; 4]]) {
        for attribute in attributes {
            for (i, word) in pack(*attribute).into_iter().enumerate() {
                gpu.write_internal(memory, renderer, REG_FIXED_ATTRIBUTE_DATA + i, word, 0xF);
            }
        }
    }

    /// halving tiled RGBA8 a tile at a time gives what averaging each
    /// pixel's four from their Morton places does.
    #[test]
    fn halving_tiles_matches_halving_pixels() {
        let (input_width, input_height, output_width) = (64u32, 32u32, 48u32);
        let mut seed = 7u32;
        let input: Vec<u8> = (0..input_width * input_height * 4)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (seed >> 16) as u8
            })
            .collect();
        let size = (32, 16);
        let mut output = vec![0xAB; (output_width * 16 * 4) as usize];
        let mut expected = output.clone();
        for y in 0..size.1 {
            for x in 0..size.0 {
                let at = morton(x, y, output_width, 4);
                for channel in 0..4 {
                    let sum: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|&(dx, dy)| input[morton(x * 2 + dx, y * 2 + dy, input_width, 4) + channel] as u32)
                        .sum();
                    expected[at + channel] = ((sum + 2) / 4) as u8;
                }
            }
        }
        halve_tiles(&input, input_width, &mut output, output_width, size);
        assert_eq!(output, expected);
    }

    #[test]
    fn immediate_mode_vertices_are_drawn() {
        let mut memory = FlatMemory(vec![0; 0x2000]);
        let mut renderer = SoftwareRenderer::default();
        let mut gpu = gpu_for_immediate_draws(&mut memory, &mut renderer);

        gpu.write_internal(&mut memory, &mut renderer, REG_FIXED_ATTRIBUTE_INDEX, 0xF, 0xF);
        let red = [1.0, 0.0, 0.0, 1.0];
        // one triangle big enough to cover the whole target.
        for position in [[-1.0, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]] {
            send_vertex(&mut gpu, &mut memory, &mut renderer, &[position, red]);
        }
        assert_eq!(gpu.draw_calls, 0, "nothing is drawn until the state moves on");

        // leaving immediate mode is an ordinary register write.
        gpu.write_internal(&mut memory, &mut renderer, 0x245, 1, 0xF);
        assert_eq!(gpu.draw_calls, 1);
        assert_eq!(gpu.vertices_drawn, 3);
        for pixel in memory.0[COLOR_BUFFER as usize..][..8 * 8 * 4].chunks(4) {
            assert_eq!(ColorFormat::Rgba8.decode(pixel), [255, 0, 0, 255]);
        }
    }

    #[test]
    fn fixed_attributes_below_fifteen_are_defaults_not_vertices() {
        let mut memory = FlatMemory(vec![0; 0x2000]);
        let mut renderer = SoftwareRenderer::default();
        let mut gpu = gpu_for_immediate_draws(&mut memory, &mut renderer);

        gpu.write_internal(&mut memory, &mut renderer, REG_FIXED_ATTRIBUTE_INDEX, 2, 0xF);
        send_vertex(&mut gpu, &mut memory, &mut renderer, &[[0.5, 0.25, 0.0, 1.0]]);
        assert_eq!(gpu.fixed_attributes[2], [0.5, 0.25, 0.0, 1.0]);
        // the index moves on to the next attribute.
        assert_eq!(gpu.internal[REG_FIXED_ATTRIBUTE_INDEX] & 0xF, 3);
        gpu.write_internal(&mut memory, &mut renderer, 0x245, 1, 0xF);
        assert_eq!(gpu.draw_calls, 0);
    }

    /// a point goes in, the geometry shader emits a triangle covering the
    /// whole target from it, the way particle systems expand their points.
    #[test]
    fn a_geometry_shader_expands_a_point() {
        let mut memory = FlatMemory(vec![0; 0x2000]);
        let mut renderer = SoftwareRenderer::default();
        let mut gpu = gpu_for_immediate_draws(&mut memory, &mut renderer);

        let mov = |dest: u32, source: u32| (0x13 << 26) | (dest << 21) | (source << 12);
        let setemit = |slot: u32, primitive: bool| (0x2B << 26) | (slot << 24) | ((primitive as u32) << 23);
        let emit = 0x2A << 26;
        let program = [
            setemit(0, false),
            mov(0, 0x20), // o0 = c0
            mov(1, 1),    // o1 = v1, the point's color
            emit,
            setemit(1, false),
            mov(0, 0x21),
            emit,
            setemit(2, true),
            mov(0, 0x22),
            emit,
            0x22 << 26,
        ];
        gpu.geometry_shader.program[..program.len()].copy_from_slice(&program);
        gpu.geometry_shader.descriptors[0] = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        gpu.geometry_shader.float_uniforms[0] = [-1.0, -1.0, 0.0, 1.0];
        gpu.geometry_shader.float_uniforms[1] = [3.0, -1.0, 0.0, 1.0];
        gpu.geometry_shader.float_uniforms[2] = [-1.0, 3.0, 0.0, 1.0];

        let mut write = |register: usize, value: u32| {
            gpu.write_internal(&mut memory, &mut renderer, register, value, 0xF)
        };
        write(REG_GEOSTAGE_CONFIG, 2);
        write(REG_VS_COM_MODE, 1);
        write(REG_VS_OUTPUT_TOTAL, 1); // two attributes per vertex
        write(REG_GS_BLOCK + SHADER_INPUT_CONFIG, 1); // one vertex per invocation
        write(REG_GS_BLOCK + SHADER_INPUT_MAP_LOW, 0x10);
        write(REG_PRIMITIVE_CONFIG, 3 << 8);
        write(REG_FIXED_ATTRIBUTE_INDEX, 0xF);

        let green = [0.0, 1.0, 0.0, 1.0];
        send_vertex(&mut gpu, &mut memory, &mut renderer, &[[0.0, 0.0, 0.0, 1.0], green]);
        gpu.write_internal(&mut memory, &mut renderer, 0x245, 1, 0xF);

        for pixel in memory.0[COLOR_BUFFER as usize..][..8 * 8 * 4].chunks(4) {
            assert_eq!(ColorFormat::Rgba8.decode(pixel), [0, 255, 0, 255]);
        }
    }

    /// unless the geometry unit is configured separately, it runs the same
    /// program the vertex shader was given.
    #[test]
    fn the_geometry_unit_mirrors_the_vertex_program_by_default() {
        let mut memory = FlatMemory(vec![0; 0x100]);
        let mut renderer = SoftwareRenderer::default();
        let mut gpu = Gpu::new();
        let mut write = |register: usize, value: u32| {
            gpu.write_internal(&mut memory, &mut renderer, register, value, 0xF)
        };
        write(REG_VS_BLOCK + SHADER_PROGRAM_INDEX, 0);
        write(REG_VS_BLOCK + SHADER_PROGRAM_DATA, 0x1234_5678);
        write(REG_VS_COM_MODE, 1);
        write(REG_VS_BLOCK + SHADER_PROGRAM_DATA, 0x9ABC_DEF0);
        assert_eq!(gpu.vertex_shader.program[..2], [0x1234_5678, 0x9ABC_DEF0]);
        assert_eq!(gpu.geometry_shader.program[..2], [0x1234_5678, 0]);
    }

    /// one register write, as a command list encodes it.
    fn command(register: usize, value: u32) -> [u32; 2] {
        [value, register as u32 | (0xF << 16)]
    }

    /// the main list jumps into a sub-buffer through channel 0, and the
    /// sub-buffer jumps back to the rest of the main list through
    /// channel 1, commands on both sides of the jump run.
    #[test]
    fn command_buffers_jump_into_each_other() {
        const MAIN: u32 = 0x100;
        const RETURN: u32 = 0x140;
        const SUB: u32 = 0x200;
        let mut memory = FlatMemory(vec![0; 0x400]);
        let mut write_words = |address: u32, words: &[u32]| {
            for (i, word) in words.iter().enumerate() {
                let at = address as usize + i * 4;
                memory.0[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        };
        let main: Vec<u32> = [
            command(REG_CMDBUF_ADDR0, SUB >> 3),
            command(REG_CMDBUF_SIZE0, 32 >> 3),
            command(REG_CMDBUF_ADDR0 + 1, RETURN >> 3),
            command(REG_CMDBUF_SIZE0 + 1, 8 >> 3),
            command(REG_CMDBUF_JUMP0, 1),
            // never reached, the jump leaves this list for good.
            command(0x0101, 0xDEAD),
        ]
        .concat();
        write_words(MAIN, &main);
        write_words(RETURN, &command(0x0102, 0x2222));
        let sub: Vec<u32> = [
            command(0x0100, 0x1111),
            command(REG_CMDBUF_JUMP1, 1),
            [0, 0],
            [0, 0],
        ]
        .concat();
        write_words(SUB, &sub);

        let mut gpu = Gpu::new();
        let mut renderer = SoftwareRenderer::default();
        gpu.process_command_list(&mut memory, &mut renderer, MAIN, main.len() as u32 * 4);
        assert_eq!(gpu.internal[0x0100], 0x1111, "the sub-buffer ran");
        assert_eq!(gpu.internal[0x0102], 0x2222, "the jump back ran");
        assert_eq!(gpu.internal[0x0101], 0, "a jump does not return");
    }

    /// flat memory that hands its bytes out in one piece, and nothing
    /// else, to see a fill go in place.
    struct WholeMemory(Vec<u8>);

    impl GpuMemory for WholeMemory {
        fn read(&mut self, _addr: u32, _out: &mut [u8]) {
            unreachable!("read in one piece")
        }

        fn write(&mut self, _addr: u32, _data: &[u8]) {
            unreachable!("written in one piece")
        }

        fn slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
            self.0.get_mut(addr as usize..addr as usize + len)
        }

        fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
            self.0.get(addr as usize..addr as usize + len)
        }
    }

    /// a fill repeats its pattern over the range, two and three byte ones
    /// too, in place where memory is in one piece and through a copy where
    /// it is not, and leaves the rest alone.
    #[test]
    fn fills_repeat_their_pattern() {
        for (width, pattern) in [(2, &[0x11u8, 0x22][..]), (3, &[0x11, 0x22, 0x33]), (4, &[0x11, 0x22, 0x33, 0x44])] {
            let mut expected = vec![0xEE; 0x200];
            for (byte, &value) in expected[0x100..0x100 + 101].iter_mut().zip(pattern.iter().cycle()) {
                *byte = value;
            }
            let mut flat = FlatMemory(vec![0xEE; 0x200]);
            Gpu::new().memory_fill(&mut flat, 0x100, 0x100 + 101, 0x4433_2211, width);
            assert_eq!(flat.0, expected, "{width} bytes through a copy");
            let mut whole = WholeMemory(vec![0xEE; 0x200]);
            Gpu::new().memory_fill(&mut whole, 0x100, 0x100 + 101, 0x4433_2211, width);
            assert_eq!(whole.0, expected, "{width} bytes in place");
        }
    }

    /// float uniforms sent in a burst, to one data register or along all
    /// of them, land as the words do one at a time.
    #[test]
    fn a_burst_of_uniforms_lands_like_single_writes() {
        let data = REG_VS_BLOCK + SHADER_UNIFORM_DATA;
        let header = |extra: u32, consecutive: bool| data as u32 | (0xF << 16) | (extra << 20) | ((consecutive as u32) << 31);
        let [a, b] = [pack([1.0, 2.0, 3.0, 4.0]), pack([5.0, 6.0, 7.0, 8.0])];
        // single floats go w first
        let wide = [12.0f32, 11.0, 10.0, 9.0].map(f32::to_bits);
        let list: Vec<u32> = [
            &command(REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 5)[..],
            &[a[0], header(5, false), a[1], a[2], b[0], b[1], b[2], 0],
            &command(REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0x8000_0009),
            &[wide[0], header(3, true), wide[1], wide[2], wide[3], 0],
        ]
        .concat();
        let mut memory = FlatMemory(vec![0; 0x200]);
        for (i, word) in list.iter().enumerate() {
            memory.0[0x100 + i * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
        let mut gpu = Gpu::new();
        let mut renderer = SoftwareRenderer::default();
        gpu.process_command_list(&mut memory, &mut renderer, 0x100, list.len() as u32 * 4);
        assert_eq!(gpu.vertex_shader.float_uniforms[5..7], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        assert_eq!(gpu.vertex_shader.float_uniforms[9], [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(gpu.internal[data..data + 4], wide);
    }

    /// a renderer that keeps every call it gets, to hold two runs against
    /// each other.
    #[derive(Default)]
    struct Recorder(Vec<Call>);

    #[derive(Debug, PartialEq)]
    enum Call {
        Draw(bool, Vec<u32>),
        Code(u32),
        Descriptor(u32),
    }

    impl Renderer for Recorder {
        fn name(&self) -> &'static str {
            "recorder"
        }

        fn draw(&mut self, call: DrawCall<'_>) {
            self.0.push(Call::Draw(call.indexed, call.registers.to_vec()));
        }

        fn upload_shader_code(&mut self, word: u32) {
            self.0.push(Call::Code(word));
        }

        fn upload_shader_operand_descriptor(&mut self, word: u32) {
            self.0.push(Call::Descriptor(word));
        }
    }

    /// the renderer hears whether a draw takes its vertices through the
    /// index buffer, which the walker below cannot tell apart from a swap.
    #[test]
    fn draw_commands_say_whether_they_are_indexed() {
        let list = [command(REG_DRAW_ARRAYS, 1), command(REG_DRAW_ELEMENTS, 1)].concat();
        let mut memory = FlatMemory(vec![0; 0x200]);
        for (i, word) in list.iter().enumerate() {
            memory.0[0x100 + i * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
        let mut gpu = Gpu::new();
        let mut renderer = Recorder::default();
        gpu.process_command_list(&mut memory, &mut renderer, 0x100, list.len() as u32 * 4);
        let indexed: Vec<bool> = renderer
            .0
            .iter()
            .filter_map(|call| match call {
                Call::Draw(indexed, _) => Some(*indexed),
                _ => None,
            })
            .collect();
        assert_eq!(indexed, [false, true]);
    }

    /// a command list run as it was before commands went in one go, a
    /// write_internal for every word, to hold run_command_list against.
    fn run_word_by_word(gpu: &mut Gpu, memory: &mut FlatMemory, renderer: &mut dyn Renderer, paddr: u32, size: u32) {
        gpu.resources.textures.begin_list();
        let mut words = Vec::new();
        read_command_buffer(memory, paddr, size, &mut words);
        let mut jumps = 0u32;
        gpu.pending_jump = None;
        let mut index = 0usize;
        while index + 1 < words.len() {
            let data = words[index];
            let header = words[index + 1];
            index += 2;
            let register = (header & 0xFFFF) as usize;
            let mask = (header >> 16) & 0xF;
            let extra = ((header >> 20) & 0xFF) as usize;
            let consecutive = header & 0x8000_0000 != 0;
            if register >= gpu.internal.len() {
                break;
            }
            gpu.write_internal(memory, renderer, register, data, mask);
            for i in 0..extra {
                if index >= words.len() {
                    break;
                }
                let value = words[index];
                index += 1;
                let target = if consecutive { register + i + 1 } else { register };
                gpu.write_internal(memory, renderer, target, value, mask);
            }
            if !extra.is_multiple_of(2) {
                index += 1;
            }
            index = index.min(words.len());
            if let Some((address, size)) = gpu.pending_jump.take() {
                jumps += 1;
                if jumps > 0x10000 {
                    break;
                }
                read_command_buffer(memory, address, size, &mut words);
                index = 0;
            }
        }
        gpu.flush_immediate(memory);
        gpu.command_lists += 1;
    }

    /// everything about a GPU a command list can change, and memory, by
    /// name, floats as their bits.
    fn state(gpu: &Gpu, memory: &FlatMemory) -> Vec<(String, Vec<u64>)> {
        fn floats<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u64> {
            values.into_iter().map(|value| value.to_bits() as u64).collect()
        }
        fn words<'a>(values: impl IntoIterator<Item = &'a u32>) -> Vec<u64> {
            values.into_iter().map(|&value| value as u64).collect()
        }
        let tables = &gpu.resources;
        let (colors, steps) = tables.proctex_tables.colors();
        let mut parts = vec![
            ("registers".to_string(), words(gpu.internal.iter())),
            ("fixed attributes".into(), floats(gpu.fixed_attributes.as_flattened())),
            (
                "fixed attribute words".into(),
                [words(&gpu.fixed_attribute_staging), vec![gpu.fixed_attribute_words as u64]].concat(),
            ),
            (
                "immediate attributes".into(),
                [floats(gpu.immediate.attributes.as_flattened()), vec![gpu.immediate.next_attribute as u64]].concat(),
            ),
            ("immediate vertices".into(), floats(gpu.immediate.vertices.iter().flat_map(|vertex| vertex.as_flattened()))),
            ("pending jump".into(), gpu.pending_jump.map_or(vec![], |(address, size)| vec![address as u64, size as u64])),
            ("counts".into(), vec![gpu.command_lists, gpu.draw_calls, gpu.vertices_drawn]),
            (
                "light tables".into(),
                [floats(tables.light_tables.entries().as_flattened().as_flattened()), vec![tables.light_tables.generation()]].concat(),
            ),
            (
                "procedural texture tables".into(),
                [
                    floats(tables.proctex_tables.maps().as_flattened().as_flattened()),
                    floats(colors.as_flattened()),
                    floats(steps.as_flattened()),
                    vec![tables.proctex_tables.generation()],
                ]
                .concat(),
            ),
            ("fog table".into(), [floats(tables.fog_table.entries().as_flattened()), vec![tables.fog_table.generation()]].concat()),
            ("memory".into(), memory.0.iter().map(|&byte| byte as u64).collect()),
        ];
        for (name, unit) in [("vertex", &gpu.vertex_shader), ("geometry", &gpu.geometry_shader)] {
            let (index, component, wide, staging, decoded, decoded_before) = unit.cursors();
            parts.extend([
                (format!("{name} program"), words(unit.program.iter())),
                (format!("{name} descriptors"), words(unit.descriptors.iter())),
                (format!("{name} float uniforms"), floats(unit.float_uniforms.as_flattened())),
                (
                    format!("{name} other uniforms"),
                    [words(&unit.int_uniforms.map(u32::from_le_bytes)), vec![unit.bool_uniforms as u64, unit.entry_point as u64]].concat(),
                ),
                (
                    format!("{name} cursors"),
                    vec![
                        unit.program_write_offset as u64,
                        unit.descriptor_write_offset as u64,
                        index as u64,
                        component as u64,
                        wide as u64,
                    ],
                ),
                (format!("{name} staged words"), words(&staging)),
                (format!("{name} programs decoded"), [decoded.into_iter().collect(), decoded_before].concat()),
            ]);
        }
        parts
    }

    /// where the lists of these tests go in memory.
    const LIST: u32 = 0x2000;

    /// runs a list both ways, from GPUs and memory set up the same, and
    /// asserts they end up the same, the renderer's calls too. the GPU
    /// that ran it in one go is left over.
    fn run_both(setup: &dyn Fn(&mut FlatMemory, &mut dyn Renderer) -> Gpu, memory: &FlatMemory, list: &[u32], context: &str) -> Gpu {
        let mut ends = Vec::new();
        let mut gpus = Vec::new();
        for word_by_word in [false, true] {
            let mut memory = memory.clone();
            let mut renderer = Recorder::default();
            let mut gpu = setup(&mut memory, &mut renderer);
            for (i, word) in list.iter().enumerate() {
                memory.0[LIST as usize + i * 4..][..4].copy_from_slice(&word.to_le_bytes());
            }
            let size = list.len() as u32 * 4;
            if word_by_word {
                run_word_by_word(&mut gpu, &mut memory, &mut renderer, LIST, size);
            } else {
                gpu.process_command_list(&mut memory, &mut renderer, LIST, size);
            }
            ends.push((state(&gpu, &memory), renderer.0));
            gpus.push(gpu);
        }
        let [(state, calls), (expected, expected_calls)] = &ends[..] else {
            unreachable!()
        };
        for ((name, ours), (_, theirs)) in state.iter().zip(expected) {
            if ours != theirs {
                let at = ours.iter().zip(theirs).position(|(a, b)| a != b).unwrap_or(ours.len().min(theirs.len()));
                panic!("{context}: {name} differ at {at:#X}, {:X?} where a word at a time gives {:X?}", ours.get(at), theirs.get(at));
            }
        }
        assert_eq!(calls, expected_calls, "{context}: the renderer's calls");
        gpus.swap_remove(0)
    }

    /// a command, its words and the padding after them, a word nothing reads.
    fn push_command(list: &mut Vec<u32>, register: usize, mask: u32, consecutive: bool, words: &[u32]) {
        let (first, rest) = words.split_first().expect("a command has a word");
        let header = register as u32 | (mask << 16) | ((rest.len() as u32) << 20) | ((consecutive as u32) << 31);
        list.extend([*first, header]);
        list.extend(rest);
        if rest.len() % 2 == 1 {
            list.push(0xA5A5_0000 | rest.len() as u32);
        }
    }

    /// a little random number generator, the same numbers every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, n: usize) -> usize {
            self.next() as usize % n
        }

        fn chance(&mut self, percent: usize) -> bool {
            self.below(100) < percent
        }

        fn words(&mut self, count: usize) -> Vec<u32> {
            (0..count).map(|_| self.next()).collect()
        }
    }

    /// where a short list for jumps to go to sits, see busy_gpu.
    const TARGET: u32 = 0x800;

    /// a value for a register that keeps draws empty, sends jumps to the
    /// short list at TARGET and keeps the fixed attributes out of
    /// immediate mode.
    fn tame(register: usize, value: u32) -> u32 {
        match register {
            REG_VERTEX_COUNT => 0,
            REG_CMDBUF_SIZE0 | 0x239 => value & 1,
            REG_CMDBUF_ADDR0 | 0x23B => TARGET >> 3,
            REG_FIXED_ATTRIBUTE_INDEX => (value & !0xF) | ((value & 0xF) % 15),
            _ => value,
        }
    }

    /// a GPU holding all sorts of state from a seed, its shader units part
    /// way through uploads, with a short list at TARGET for jumps.
    fn busy_gpu(seed: u64, memory: &mut FlatMemory) -> Gpu {
        let mut random = Random(seed);
        memory.0[TARGET as usize..][..8].copy_from_slice(&[0x5555_5555u32, 0x000F_0010].map(u32::to_le_bytes).concat());
        let mut gpu = Gpu::new();
        for register in 0..INTERNAL_REGISTER_WORDS {
            gpu.internal[register] = tame(register, random.next());
        }
        for unit in [&mut gpu.vertex_shader, &mut gpu.geometry_shader] {
            for word in unit.program.iter_mut().chain(unit.descriptors.iter_mut()) {
                *word = random.next();
            }
            unit.program_write_offset = random.below(shader::PROGRAM_SIZE + 4);
            unit.descriptor_write_offset = random.below(shader::DESCRIPTOR_SIZE + 4);
            unit.set_float_uniform_index(random.next());
            for _ in 0..random.below(4) {
                unit.upload_float_uniform(random.next());
            }
        }
        gpu
    }

    /// every register, with every shape of command, alone, in bursts, in
    /// runs on through the ones after it, with masks, gives what writing
    /// each word on its own gives, from a GPU holding state.
    #[test]
    fn every_register_takes_commands_as_single_writes() {
        // mask, extra words, consecutive
        let shapes = [
            (0xF, 0, false),
            (0x9, 0, false),
            (0xF, 3, false),
            (0x6, 2, false),
            (0x0, 1, false),
            (0xF, 4, true),
            (0xF, 9, true),
            (0xA, 5, true),
            (0x0, 2, true),
        ];
        let mut random = Random(1);
        for register in 0..INTERNAL_REGISTER_WORDS {
            for (shape, &(mask, extra, consecutive)) in shapes.iter().enumerate() {
                let seed = random.next() as u64;
                let words: Vec<u32> =
                    (0..=extra).map(|i| tame(if consecutive { register + i } else { register }, random.next())).collect();
                let mut list = Vec::new();
                push_command(&mut list, register, mask, consecutive, &words);
                let memory = FlatMemory(vec![0; 0x3000]);
                run_both(&|memory, _| busy_gpu(seed, memory), &memory, &list, &format!("register {register:#05X}, shape {shape}"));
            }
        }
    }

    /// a word for the program port that keeps the program harmless, moves
    /// into the first outputs, no-ops and ends.
    fn instruction(random: &mut Random) -> u32 {
        match random.below(4) {
            0 => 0x21 << 26,
            1 => 0x22 << 26,
            _ => (0x13 << 26) | ((random.below(4) as u32) << 21) | ((random.below(0x80) as u32) << 12),
        }
    }

    /// words for a run of registers in a shader unit's block from an
    /// offset, harmless instructions where they land on the program port.
    fn block_words(random: &mut Random, offset: usize, count: usize, consecutive: bool) -> Vec<u32> {
        (0..count)
            .map(|i| {
                let offset = if consecutive { offset + i } else { offset };
                if (SHADER_PROGRAM_DATA..=SHADER_PROGRAM_DATA_END).contains(&offset) {
                    instruction(random)
                } else {
                    random.next()
                }
            })
            .collect()
    }

    /// a command list of the shapes titles send, the ones that go in one
    /// go and others, with immediate-mode vertices between them.
    fn random_list(random: &mut Random) -> Vec<u32> {
        let mut list = Vec::new();
        for _ in 0..1 + random.below(24) {
            let mask = if random.chance(80) { 0xF } else { random.below(16) as u32 };
            let block = if random.chance(70) { REG_VS_BLOCK } else { REG_GS_BLOCK };
            let consecutive = random.chance(40);
            match random.below(12) {
                // state no draw here reads, alone, in bursts and in runs,
                // some of them past the last register
                0 => {
                    let (start, end) = [(0x000, 0x03F), (0x200, 0x227), (0x2DE, 0x2FF)][random.below(3)];
                    let register = start + random.below(end - start + 1);
                    let room = if consecutive && end != 0x2FF { end - register } else { 40 };
                    let count = 1 + random.below(room + 1);
                    let words = random.words(count);
                    push_command(&mut list, register, mask, consecutive, &words);
                }
                // float uniforms, the index and then the data, which can
                // run on past the data registers
                1 => {
                    let index = random.next() & 0x8000_007F;
                    push_command(&mut list, block + SHADER_UNIFORM_INDEX, 0xF, false, &[index]);
                    let offset = SHADER_UNIFORM_DATA + random.below(8);
                    let count = if consecutive { 1 + random.below(SHADER_PROGRAM_DATA_END - offset + 1) } else { 1 + random.below(70) };
                    let words = block_words(random, offset, count, consecutive);
                    push_command(&mut list, block + offset, mask, consecutive, &words);
                }
                // from the index on into the data, and past it
                2 => {
                    let count = 1 + random.below(14);
                    let words = block_words(random, SHADER_UNIFORM_INDEX, count, true);
                    push_command(&mut list, block + SHADER_UNIFORM_INDEX, mask, true, &words);
                }
                // several indices in a row, the last one counts
                3 => {
                    let count = 1 + random.below(4);
                    let words = random.words(count);
                    push_command(&mut list, block + SHADER_UNIFORM_INDEX, mask, false, &words);
                }
                // a program, mirrored to the geometry unit or not
                4 => {
                    push_command(&mut list, REG_VS_COM_MODE, 0xF, false, &[random.next() & 1]);
                    let start = random.next() & 0xFFF;
                    push_command(&mut list, block + SHADER_PROGRAM_INDEX, 0xF, false, &[start]);
                    let offset = SHADER_PROGRAM_DATA + random.below(8);
                    let count = if consecutive { 1 + random.below(SHADER_PROGRAM_DATA_END - offset + 1) } else { 1 + random.below(90) };
                    let words: Vec<u32> = (0..count).map(|_| instruction(random)).collect();
                    push_command(&mut list, block + offset, mask, consecutive, &words);
                }
                // operand descriptors the same way, running on past them
                5 => {
                    push_command(&mut list, REG_VS_COM_MODE, 0xF, false, &[random.next() & 1]);
                    let start = random.next() & 0x7F;
                    push_command(&mut list, block + SHADER_DESCRIPTOR_INDEX, 0xF, false, &[start]);
                    let offset = SHADER_DESCRIPTOR_DATA + random.below(8);
                    let count = if consecutive { 1 + random.below(SHADER_BLOCK_SIZE + 3 - offset) } else { 1 + random.below(40) };
                    push_command(&mut list, block + offset, mask, consecutive, &random.words(count));
                }
                // one of the tables, its index and then its data
                6 => {
                    let (index, data) = [
                        (lighting::REG_TABLE_INDEX, lighting::REG_TABLE_DATA),
                        (proctex::REG_TABLE_INDEX, proctex::REG_TABLE_DATA),
                        (fog::REG_TABLE_INDEX, fog::REG_TABLE_DATA),
                    ][random.below(3)];
                    push_command(&mut list, index, 0xF, false, &[random.next() & 0x1FFF]);
                    let offset = random.below(8);
                    let count = if consecutive { 1 + random.below(8 - offset) } else { 1 + random.below(130) };
                    push_command(&mut list, data + offset, mask, consecutive, &random.words(count));
                }
                // the boolean and integer uniforms, and the entry point
                7 => {
                    if random.chance(50) {
                        let count = 1 + random.below(5);
                        let words = random.words(count);
                        push_command(&mut list, block + SHADER_BOOL_UNIFORMS, mask, true, &words);
                    } else {
                        push_command(&mut list, block + SHADER_ENTRY_POINT, mask, false, &[random.next()]);
                    }
                }
                // fixed attributes, the index and then three words each
                8 => {
                    let mut words = vec![random.below(15) as u32];
                    for _ in 0..random.below(4) {
                        let value = [0.5, -2.0, 0.25, 1.0].map(|c: f32| c * (1 + random.below(8)) as f32);
                        words.extend(pack(value));
                    }
                    if consecutive && words.len() <= 4 {
                        push_command(&mut list, REG_FIXED_ATTRIBUTE_INDEX, mask, true, &words);
                    } else {
                        push_command(&mut list, REG_FIXED_ATTRIBUTE_INDEX, 0xF, false, &words[..1]);
                        if words.len() > 1 {
                            push_command(&mut list, REG_FIXED_ATTRIBUTE_DATA, mask, false, &words[1..]);
                        }
                    }
                }
                // draws, of no vertices, the renderer still hears of them
                9 => {
                    let register = if random.chance(50) { REG_DRAW_ARRAYS } else { REG_DRAW_ELEMENTS };
                    push_command(&mut list, register, mask, false, &[random.next()]);
                }
                // state running on into a shader unit's uniforms
                10 => {
                    let count = 1 + random.below(9);
                    let words = random.words(count);
                    push_command(&mut list, REG_VS_BLOCK - 2, mask, true, &words);
                }
                // a uniform upload with the data in single commands
                _ => {
                    push_command(&mut list, block + SHADER_UNIFORM_INDEX, 0xF, false, &[random.next() & 0x8000_007F]);
                    for _ in 0..random.below(9) {
                        push_command(&mut list, block + SHADER_UNIFORM_DATA + random.below(8), mask, false, &[random.next()]);
                    }
                }
            }
            // vertices sent one attribute at a time, drawn before whatever
            // comes next, a vertex left half sent now and then
            if random.chance(30) {
                let mut words = Vec::new();
                for _ in 0..1 + random.below(6) {
                    let position = [random.below(5) as f32 - 2.0, random.below(5) as f32 - 2.0, 0.5, 1.0];
                    let color = [random.below(2) as f32, random.below(2) as f32, 1.0, 1.0];
                    words.extend(pack(position));
                    words.extend(pack(color));
                }
                words.truncate(words.len() - if random.chance(20) { random.below(6) } else { 0 });
                // without the index again they join the vertices before
                // unless the command between drew those
                if random.chance(50) {
                    push_command(&mut list, REG_FIXED_ATTRIBUTE_INDEX, 0xF, false, &[0xF]);
                }
                push_command(&mut list, REG_FIXED_ATTRIBUTE_DATA, 0xF, false, &words);
            }
        }
        list
    }

    /// lists of random commands of every shape, with immediate-mode
    /// vertices between them, whole and cut short anywhere, give what
    /// writing each word on its own gives.
    #[test]
    fn random_lists_run_as_single_writes() {
        let mut random = Random(2);
        for case in 0..400 {
            let list = random_list(&mut random);
            let memory = FlatMemory(vec![0; 0x10000]);
            let setup = |memory: &mut FlatMemory, renderer: &mut dyn Renderer| gpu_for_immediate_draws(memory, renderer);
            run_both(&setup, &memory, &list, &format!("list {case}"));
            let cut = random.below(list.len() + 1);
            run_both(&setup, &memory, &list[..cut], &format!("list {case} cut to {cut} words"));
        }
    }

    /// a list with every shape that goes in one go, cut after each word,
    /// ends as it does a word at a time.
    #[test]
    fn a_list_cut_anywhere_runs_as_single_writes() {
        let mut list = Vec::new();
        push_command(&mut list, 0x0010, 0xF, true, &[1, 2, 3, 4]);
        push_command(&mut list, 0x0020, 0x3, false, &[5, 6, 7]);
        push_command(&mut list, REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, true, &[0x8000_0002, 1, 2, 3, 4, 5]);
        push_command(&mut list, REG_VS_BLOCK + SHADER_UNIFORM_DATA, 0xF, false, &[6, 7, 8, 9, 10, 11, 12]);
        push_command(&mut list, REG_GS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, false, &[3, 4]);
        push_command(&mut list, REG_GS_BLOCK + SHADER_UNIFORM_DATA + 2, 0xF, true, &[13, 14, 15, 16]);
        push_command(&mut list, REG_VS_BLOCK + SHADER_PROGRAM_DATA, 0xF, false, &[0x21 << 26, 0x22 << 26, 0x21 << 26]);
        push_command(&mut list, REG_VS_BLOCK + SHADER_DESCRIPTOR_DATA + 1, 0xF, true, &[17, 18, 19]);
        push_command(&mut list, lighting::REG_TABLE_DATA, 0xF, false, &[20, 21, 22, 23, 24]);
        push_command(&mut list, proctex::REG_TABLE_DATA + 3, 0xF, true, &[25, 26, 27]);
        push_command(&mut list, fog::REG_TABLE_DATA, 0xF, false, &[28, 29]);
        push_command(&mut list, 0x02FE, 0xF, true, &[30, 31, 32, 33]);
        for cut in 0..=list.len() {
            let memory = FlatMemory(vec![0; 0x3000]);
            run_both(&|memory, _| busy_gpu(3, memory), &memory, &list[..cut], &format!("cut to {cut} words"));
        }
    }

    /// vertices sent in immediate mode are drawn before each shape of
    /// command that goes in one go, not with the ones sent after it.
    #[test]
    fn every_shape_draws_the_vertices_waiting_first() {
        let mov = 0x13 << 26;
        let identity = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        // register, mask, consecutive and the words of a command that
        // leaves the drawing as it was
        let shapes: [(usize, u32, bool, &[u32]); 17] = [
            (0x0010, 0xF, false, &[1]),
            (0x0010, 0x0, false, &[1]),
            (0x0010, 0x3, false, &[1, 2, 3]),
            (0x0010, 0xF, true, &[1, 2, 3]),
            // state on into a table
            (lighting::REG_TABLE_DATA - 2, 0xF, true, &[1, 2, 3]),
            (REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, false, &[2]),
            (REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, false, &[2, 3]),
            (REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, true, &[2, 1, 2, 3]),
            (REG_VS_BLOCK + SHADER_UNIFORM_DATA, 0xF, false, &[1, 2, 3]),
            (REG_GS_BLOCK + SHADER_UNIFORM_DATA + 1, 0xF, true, &[1, 2, 3]),
            (REG_VS_BLOCK + SHADER_PROGRAM_DATA, 0xF, false, &[mov]),
            (REG_GS_BLOCK + SHADER_PROGRAM_DATA, 0xF, false, &[mov, mov]),
            (REG_VS_BLOCK + SHADER_DESCRIPTOR_DATA, 0xF, false, &[identity]),
            (REG_GS_BLOCK + SHADER_DESCRIPTOR_DATA + 2, 0xF, true, &[1, 2]),
            (lighting::REG_TABLE_DATA, 0xF, false, &[1, 2]),
            (proctex::REG_TABLE_DATA + 1, 0xF, true, &[1, 2]),
            (fog::REG_TABLE_DATA, 0xF, false, &[1]),
        ];
        let triangle = |red: f32| {
            let corners = [[-1.0, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]];
            corners.iter().flat_map(|&corner| [pack(corner), pack([red, 0.0, 1.0, 1.0])]).flatten().collect::<Vec<u32>>()
        };
        for (register, mask, consecutive, words) in shapes {
            let mut list = Vec::new();
            push_command(&mut list, REG_FIXED_ATTRIBUTE_INDEX, 0xF, false, &[0xF]);
            push_command(&mut list, REG_FIXED_ATTRIBUTE_DATA, 0xF, false, &triangle(1.0));
            push_command(&mut list, register, mask, consecutive, words);
            push_command(&mut list, REG_FIXED_ATTRIBUTE_DATA, 0xF, false, &triangle(0.0));
            let memory = FlatMemory(vec![0; 0x3000]);
            let setup = |memory: &mut FlatMemory, renderer: &mut dyn Renderer| gpu_for_immediate_draws(memory, renderer);
            let context = format!("register {register:#05X}");
            let gpu = run_both(&setup, &memory, &list, &context);
            assert_eq!((gpu.draw_calls, gpu.vertices_drawn), (2, 6), "{context}");
        }
    }

    /// a run of state ending at a jump, the way titles call their models,
    /// takes the jump, and commands of every shape run on both sides of it.
    #[test]
    fn a_run_of_state_ending_in_a_jump_takes_it() {
        const SUB: u32 = 0x2800;
        const RETURN: u32 = 0x2C00;
        let mut sub = Vec::new();
        push_command(&mut sub, REG_VS_BLOCK + SHADER_UNIFORM_INDEX, 0xF, true, &[0x8000_0004, 1, 2, 3, 4]);
        push_command(&mut sub, 0x0030, 0xF, true, &[5, 6, 7]);
        // jumps back through channel 1, a run of state on the way
        push_command(&mut sub, REG_CMDBUF_SIZE0, 0xF, true, &[0, 1, 0, RETURN >> 3, 0, 1]);
        push_command(&mut sub, 0x0031, 0xF, false, &[0xDEAD]);
        let mut back = Vec::new();
        push_command(&mut back, 0x0032, 0xF, true, &[8]);
        let mut list = Vec::new();
        push_command(&mut list, 0x0020, 0xF, true, &[9, 10, 11]);
        let size = (sub.len() as u32 * 4).div_ceil(8);
        push_command(&mut list, REG_CMDBUF_SIZE0, 0xF, true, &[size, 0, SUB >> 3, 0, 1]);
        push_command(&mut list, 0x0033, 0xF, false, &[0xDEAD]);
        let mut memory = FlatMemory(vec![0; 0x3000]);
        for (address, words) in [(SUB, &sub), (RETURN, &back)] {
            for (i, word) in words.iter().enumerate() {
                memory.0[address as usize + i * 4..][..4].copy_from_slice(&word.to_le_bytes());
            }
        }
        run_both(&|_, _| Gpu::new(), &memory, &list, "jumps");
        let mut gpu = Gpu::new();
        let mut renderer = SoftwareRenderer::default();
        for (i, word) in list.iter().enumerate() {
            memory.0[LIST as usize + i * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
        gpu.process_command_list(&mut memory, &mut renderer, LIST, list.len() as u32 * 4);
        assert_eq!(gpu.internal[0x20..0x23], [9, 10, 11]);
        // the jumps leave the rest of the lists alone
        assert_eq!(gpu.internal[0x30..0x34], [5, 6, 8, 0], "both lists ran");
        assert_eq!(gpu.vertex_shader.float_uniforms[4].map(f32::to_bits), [4, 3, 2, 1]);
    }

    /// the byte masks a command can carry, a set bit for each byte written.
    #[test]
    fn byte_masks_cover_the_bytes_their_bits_name() {
        for (mask, bits) in BYTE_MASKS.iter().enumerate() {
            for byte in 0..4 {
                assert_eq!(bits >> (byte * 8) & 0xFF, if mask & (1 << byte) != 0 { 0xFF } else { 0 }, "mask {mask:X}");
            }
        }
    }

    #[test]
    fn a_downscaling_transfer_averages_each_block() {
        let mut memory = FlatMemory(vec![0; 0x200]);
        let reds = [[0u8, 100, 200, 0], [100, 0, 0, 200]];
        for (y, row) in reds.iter().enumerate() {
            for (x, red) in row.iter().enumerate() {
                let at = 0x100 + (y * 4 + x) * 4;
                ColorFormat::Rgba8.encode([*red, 0, 0, 255], &mut memory.0[at..at + 4]);
            }
        }
        let mut gpu = Gpu::new();
        // linear in and out, RGBA8, 2x2 downscale, the output's size counts
        // input pixels
        let flags = (1 << 1) | (1 << 5) | (2 << 24);
        gpu.display_transfer(&mut memory, 0x100, 0x180, 4 | (2 << 16), 4 | (2 << 16), flags);
        let red_at = |i: usize| ColorFormat::Rgba8.decode(&memory.0[0x180 + i * 4..0x184 + i * 4])[0];
        assert_eq!([red_at(0), red_at(1)], [50, 100]);
    }

    /// a flipped transfer cropping its input's lines writes its output as
    /// many bytes later as the cropped lines take but the last, the address
    /// Tales of the Abyss gives is that far before its framebuffer.
    #[test]
    fn a_flipped_cropped_transfer_starts_its_output_later() {
        let mut memory = FlatMemory(vec![0; 0x400]);
        // a 4x2 linear RGBA8 input, its first line red, its second green
        for x in 0..4 {
            ColorFormat::Rgba8.encode([255, 0, 0, 255], &mut memory.0[0x100 + x * 4..0x104 + x * 4]);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut memory.0[0x110 + x * 4..0x114 + x * 4]);
        }
        let mut gpu = Gpu::new();
        // linear in and out, cropped to two pixels a line, flipped, RGB8 out
        let flags = 1 | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 12);
        gpu.display_transfer(&mut memory, 0x100, 0x200, 4 | (2 << 16), 2 | (2 << 16), flags);
        // two pixels cropped off one line before the last, three bytes each
        let pixel = |at: usize| ColorFormat::Rgb8.decode(&memory.0[at..at + 3]);
        let start = 0x200 + 2 * 3;
        assert_eq!(pixel(start), [0, 255, 0, 255], "the flipped output begins with the input's last line");
        assert_eq!(pixel(start + 2 * 3), [255, 0, 0, 255]);
        assert_eq!(&memory.0[0x200..start], &[0; 6], "nothing before where it starts");
    }
}
