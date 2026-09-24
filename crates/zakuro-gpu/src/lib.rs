//! PICA200 GPU emulation.

pub mod backend;
pub mod blend;
pub mod format;
pub mod raster;
pub mod registers;
pub mod renderer;
pub mod shader;
pub mod tev;
pub mod texture;

use format::ColorFormat;
use registers::*;
pub use backend::{layout, PresentError, Presenter, ScreenImage, Viewport};
pub use renderer::{DrawCall, Renderer, RendererKind, SoftwareRenderer};

/// how the GPU reaches guest memory.
pub trait GpuMemory {
    fn read(&mut self, addr: u32, out: &mut [u8]);
    fn write(&mut self, addr: u32, data: &[u8]);

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

        let pattern: Vec<u8> = match width {
            2 => value.to_le_bytes()[..2].to_vec(),
            3 => value.to_le_bytes()[..3].to_vec(),
            _ => value.to_le_bytes().to_vec(),
        };

        let mut buffer = Vec::with_capacity(length);
        while buffer.len() < length {
            let remaining = length - buffer.len();
            buffer.extend_from_slice(&pattern[..pattern.len().min(remaining)]);
        }
        log::debug!(
            "memory fill: 0x{start:08X}..0x{end:08X} with 0x{value:08X} ({width}-byte pattern)"
        );
        memory.write(address, &buffer);
        self.fills += 1;
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
        let output_width = output_dimensions & 0xFFFF;
        let output_height = output_dimensions >> 16;

        if input_width == 0 || input_height == 0 || output_width == 0 || output_height == 0 {
            return;
        }

        // flag layout, from the transfer engine's register,
        //   bit 0      flip the input vertically
        //   bit 1      the input is linear rather than tiled
        //   bit 3      raw copy, no format conversion
        //   bits 8-10  input color format
        //   bits 12-14 output color format
        //   bit 16     the output is tiled rather than linear
        //   bits 24-25 downscale
        let flip_vertically = flags & 1 != 0;
        let input_linear = flags & (1 << 1) != 0;
        let input_format = ColorFormat::from_raw((flags >> 8) & 7);
        let output_format = ColorFormat::from_raw((flags >> 12) & 7);
        let output_tiled = flags & (1 << 16) != 0;
        let downscale = (flags >> 24) & 3;

        let input_bpp = input_format.bytes_per_pixel();
        let output_bpp = output_format.bytes_per_pixel();

        let (scale_x, scale_y) = match downscale {
            1 => (2u32, 1u32),
            2 => (2, 2),
            _ => (1, 1),
        };

        let copy_width = output_width.min(input_width / scale_x);
        let copy_height = output_height.min(input_height / scale_y);

        log::debug!(
            "display transfer: 0x{input_paddr:08X} {input_width}x{input_height} \
             {input_format:?} {} -> 0x{output_paddr:08X} {output_width}x{output_height} \
             {output_format:?} {} (flags 0x{flags:08X})",
            if input_linear { "linear" } else { "tiled" },
            if output_tiled { "tiled" } else { "linear" },
        );

        let input_base = memory.translate(input_paddr);
        let output_base = memory.translate(output_paddr);

        let mut input = vec![0u8; (input_width * input_height) as usize * input_bpp];
        memory.read(input_base, &mut input);
        let mut output = vec![0u8; (output_width * output_height) as usize * output_bpp];
        // preserve whatever was already there outside the copied rectangle.
        memory.read(output_base, &mut output);

        let trace_pixels = std::env::var("ZAKURO_TRACE_PIXELS").is_ok();
        let mut distinct = std::collections::HashSet::new();

        for y in 0..copy_height {
            for x in 0..copy_width {
                let dst = if output_tiled {
                    morton(x, y, output_width, output_bpp)
                } else {
                    (y * output_width + x) as usize * output_bpp
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
                        let mut src_y = y * scale_y + dy;
                        if flip_vertically {
                            src_y = input_height.saturating_sub(1 + src_y);
                        }
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
        let output_width = (output_gap & 0xFFFF) * 16;
        let output_skip = (output_gap >> 16) * 16;

        log::debug!(
            "texture copy: 0x{input_paddr:08X} -> 0x{output_paddr:08X} size 0x{size:X}"
        );

        let mut src = memory.translate(input_paddr);
        let mut dst = memory.translate(output_paddr);

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
        self.run_command_list(memory, renderer, paddr, size);
        self.busy += start.elapsed();
    }

    fn run_command_list<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        renderer: &mut dyn Renderer,
        paddr: u32,
        size: u32,
    ) {
        let mut words = read_command_buffer(memory, paddr, size);
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

            self.write_internal(memory, renderer, register, data, mask);

            for i in 0..extra {
                if index >= words.len() {
                    break;
                }
                let value = words[index];
                index += 1;
                let target = if consecutive { register + i + 1 } else { register };
                self.write_internal(memory, renderer, target, value, mask);
            }

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
                words = read_command_buffer(memory, address, size);
                index = 0;
            }
        }

        self.flush_immediate(memory);
        self.command_lists += 1;
    }

    /// draws any immediate-mode vertices still waiting.
    fn flush_immediate<M: GpuMemory>(&mut self, memory: &mut M) {
        if self.immediate.vertices.is_empty() {
            return;
        }
        let vertices = std::mem::take(&mut self.immediate.vertices);
        self.draw_calls += 1;
        self.vertices_drawn += vertices.len() as u64;
        raster::draw_immediate(
            &self.internal,
            &self.vertex_shader,
            &self.geometry_shader,
            memory,
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
        let mut write_mask = 0u32;
        for byte in 0..4 {
            if mask & (1 << byte) != 0 {
                write_mask |= 0xFF << (byte * 8);
            }
        }
        let old = self.internal[register];
        let new = (old & !write_mask) | (value & write_mask);
        self.internal[register] = new;

        match register {
            REG_DRAW_ARRAYS | REG_DRAW_ELEMENTS => {
                self.draw_calls += 1;
                let indexed = register == REG_DRAW_ELEMENTS;
                renderer.draw(DrawCall {
                    indexed,
                    registers: &self.internal,
                });
                let vertices =
                    raster::draw(
                    &self.internal,
                    &self.vertex_shader,
                    &self.geometry_shader,
                    &self.fixed_attributes,
                    memory,
                    indexed,
                );
                self.vertices_drawn += vertices as u64;
            }

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

/// reads a command buffer as words.
fn read_command_buffer<M: GpuMemory>(memory: &mut M, paddr: u32, size: u32) -> Vec<u32> {
    let mut buffer = vec![0u8; size as usize];
    memory.read(memory.translate(paddr), &mut buffer);
    buffer
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

const REG_GS_BLOCK_END: usize = REG_GS_BLOCK + SHADER_BLOCK_SIZE - 1;
const REG_VS_BLOCK_END: usize = REG_VS_BLOCK + SHADER_BLOCK_SIZE - 1;

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
        let exponent = ((bits >> 23) & 0xFF) - 127 + 63;
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
    fn gpu_for_immediate_draws(memory: &mut FlatMemory, renderer: &mut SoftwareRenderer) -> Gpu {
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
        gpu
    }

    fn send_vertex(gpu: &mut Gpu, memory: &mut FlatMemory, renderer: &mut SoftwareRenderer, attributes: &[[f32; 4]]) {
        for attribute in attributes {
            for (i, word) in pack(*attribute).into_iter().enumerate() {
                gpu.write_internal(memory, renderer, REG_FIXED_ATTRIBUTE_DATA + i, word, 0xF);
            }
        }
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
        // linear in and out, RGBA8, 2x2 downscale
        let flags = (1 << 1) | (2 << 24);
        gpu.display_transfer(&mut memory, 0x100, 0x180, 4 | (2 << 16), 2 | (1 << 16), flags);
        let red_at = |i: usize| ColorFormat::Rgba8.decode(&memory.0[0x180 + i * 4..0x184 + i * 4])[0];
        assert_eq!([red_at(0), red_at(1)], [50, 100]);
    }
}
