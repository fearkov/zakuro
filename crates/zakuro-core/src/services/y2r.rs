//! y2r:u, the hardware that turns YUV video frames into RGB images, which
//! titles play movies through. a conversion runs as soon as it starts and
//! signals its end at once, the arithmetic being the hardware's own as
//! Citra worked it out.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::ObjectId;
use crate::kernel::sync::ResetType;
use crate::System;

/// the coefficients of the four standard conversions, ITU-R BT.601 and 709,
/// full range and scaled.
const STANDARD: [[i32; 8]; 4] = [
    [0x100, 0x166, 0xB6, 0x58, 0x1C5, -0x166F, 0x10EE, -0x1C5B],
    [0x100, 0x193, 0x77, 0x2F, 0x1DB, -0x1933, 0xA7C, -0x1D51],
    [0x12A, 0x198, 0xD0, 0x64, 0x204, -0x1BDE, 0x10F2, -0x229B],
    [0x12A, 0x1CA, 0x88, 0x36, 0x21C, -0x1F04, 0x99C, -0x2421],
];

/// a buffer a conversion reads from or writes to, in pieces of transfer
/// bytes with gap bytes skipped after each.
#[derive(Debug, Clone, Copy, Default)]
struct Buffer {
    address: u32,
    transfer: u16,
    gap: u16,
}

pub struct Y2rState {
    input_format: u32,
    output_format: u32,
    rotation: u32,
    /// the output in 8x8 tiles, the way textures are, rather than in rows.
    tiled: bool,
    line_width: u32,
    lines: u32,
    coefficients: [i32; 8],
    standard: u32,
    alpha: u32,
    dithering: [u32; 3],
    end_interrupt: bool,
    y: Buffer,
    u: Buffer,
    v: Buffer,
    yuyv: Buffer,
    output: Buffer,
    /// signalled at the end of each conversion.
    event: Option<ObjectId>,
    scratch: Scratch,
}

/// what a conversion works in, kept from one to the next, a movie
/// converting a frame at a time.
#[derive(Default)]
struct Scratch {
    /// the samples gathered, Y, U and V, or the YUYV stream in the first.
    planes: [Vec<u8>; 3],
    /// a transfer's worth of 16 bit samples.
    piece: Vec<u8>,
    /// a YUYV line taken apart into its Y, U and V.
    line: [Vec<u8>; 3],
    /// the chroma terms of a line, red, green and blue, one a pixel.
    chroma: [Vec<i32>; 3],
    /// eight lines of pixels in the output format.
    pixels: Vec<u32>,
    /// the image as it goes to the output buffer.
    out: Vec<u8>,
}

impl Default for Y2rState {
    fn default() -> Self {
        Y2rState {
            input_format: 0,
            output_format: 0,
            rotation: 0,
            tiled: false,
            line_width: 0,
            lines: 0,
            coefficients: STANDARD[0],
            standard: 0,
            alpha: 0xFF,
            dithering: [0; 3],
            end_interrupt: false,
            y: Buffer::default(),
            u: Buffer::default(),
            v: Buffer::default(),
            yuyv: Buffer::default(),
            output: Buffer::default(),
            event: None,
            scratch: Scratch::default(),
        }
    }
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);
    let reply = |system: &mut System, values: &[u32]| buffer.reply(&mut system.memory, command, values);
    match command {
        // SetInputFormat / GetInputFormat
        0x0001 => {
            state(system).input_format = word(system, 1) & 0xFF;
            reply(system, &[]);
        }
        0x0002 => {
            let value = state(system).input_format;
            reply(system, &[value]);
        }
        // SetOutputFormat / GetOutputFormat
        0x0003 => {
            state(system).output_format = word(system, 1) & 0xFF;
            reply(system, &[]);
        }
        0x0004 => {
            let value = state(system).output_format;
            reply(system, &[value]);
        }
        // SetRotation / GetRotation
        0x0005 => {
            state(system).rotation = word(system, 1) & 0xFF;
            reply(system, &[]);
        }
        0x0006 => {
            let value = state(system).rotation;
            reply(system, &[value]);
        }
        // SetBlockAlignment / GetBlockAlignment
        0x0007 => {
            state(system).tiled = word(system, 1) & 0xFF != 0;
            reply(system, &[]);
        }
        0x0008 => {
            let value = state(system).tiled as u32;
            reply(system, &[value]);
        }
        // SetSpacialDithering / SetTemporalDithering and their getters
        0x0009 | 0x000B => {
            state(system).dithering[(command as usize - 0x0009) / 2] = word(system, 1) & 0xFF;
            reply(system, &[]);
        }
        0x000A | 0x000C => {
            let value = state(system).dithering[(command as usize - 0x000A) / 2];
            reply(system, &[value]);
        }
        // SetTransferEndInterrupt / GetTransferEndInterrupt
        0x000D => {
            state(system).end_interrupt = word(system, 1) & 0xFF != 0;
            reply(system, &[]);
        }
        0x000E => {
            let value = state(system).end_interrupt as u32;
            reply(system, &[value]);
        }
        // GetTransferEndEvent
        0x000F => {
            let object = event(system);
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "y2r:u end event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        // SetSendingY / U / V / YUYV(address, image size, transfer unit,
        // gap, process), SetReceiving the same for the output
        0x0010..=0x0013 | 0x0018 => {
            let target = Buffer {
                address: word(system, 1),
                transfer: word(system, 3) as u16,
                gap: word(system, 4) as u16,
            };
            let y2r = state(system);
            match command {
                0x0010 => y2r.y = target,
                0x0011 => y2r.u = target,
                0x0012 => y2r.v = target,
                0x0013 => y2r.yuyv = target,
                _ => y2r.output = target,
            }
            reply(system, &[]);
        }
        // IsFinishedSendingYuv / Y / U / V, IsFinishedReceiving, every
        // conversion being done as soon as it starts
        0x0014..=0x0017 | 0x0019 => reply(system, &[1]),
        // SetInputLineWidth / GetInputLineWidth
        0x001A => {
            state(system).line_width = word(system, 1) & 0xFFFF;
            reply(system, &[]);
        }
        0x001B => {
            let value = state(system).line_width;
            reply(system, &[value]);
        }
        // SetInputLines / GetInputLines
        0x001C => {
            state(system).lines = word(system, 1) & 0xFFFF;
            reply(system, &[]);
        }
        0x001D => {
            let value = state(system).lines;
            reply(system, &[value]);
        }
        // SetCoefficient(eight halfwords) / GetCoefficient
        0x001E => {
            let words = [word(system, 1), word(system, 2), word(system, 3), word(system, 4)];
            let halves: [i32; 8] = std::array::from_fn(|i| (words[i / 2] >> (16 * (i % 2))) as u16 as i16 as i32);
            state(system).coefficients = halves;
            reply(system, &[]);
        }
        0x001F => {
            let c = state(system).coefficients.map(|c| c as u16 as u32);
            reply(system, &[c[0] | c[1] << 16, c[2] | c[3] << 16, c[4] | c[5] << 16, c[6] | c[7] << 16]);
        }
        // SetStandardCoefficient / GetStandardCoefficient(index)
        0x0020 => {
            let index = word(system, 1) & 0xFF;
            let y2r = state(system);
            y2r.standard = index;
            y2r.coefficients = STANDARD[index as usize % 4];
            reply(system, &[]);
        }
        0x0021 => {
            let c = STANDARD[(word(system, 1) & 3) as usize].map(|c| c as u16 as u32);
            reply(system, &[c[0] | c[1] << 16, c[2] | c[3] << 16, c[4] | c[5] << 16, c[6] | c[7] << 16]);
        }
        // SetAlpha / GetAlpha
        0x0022 => {
            state(system).alpha = word(system, 1) & 0xFFFF;
            reply(system, &[]);
        }
        0x0023 => {
            let value = state(system).alpha;
            reply(system, &[value]);
        }
        // SetDitheringWeightParams / GetDitheringWeightParams, which only
        // change what dithering does, and dithering is not done
        0x0024 => reply(system, &[]),
        0x0025 => reply(system, &[0; 8]),
        // StartConversion
        0x0026 => {
            convert(system);
            let object = event(system);
            system.kernel.signal_event(object);
            reply(system, &[]);
        }
        // StopConversion, IsBusyConversion, PingProcess
        0x0027 => reply(system, &[]),
        0x0028 => reply(system, &[0]),
        0x002A => reply(system, &[0]),
        // SetPackageParameter(formats, rotation and alignment, line width
        // and lines, coefficient and alpha)
        0x0029 => {
            let (first, second, third) = (word(system, 1), word(system, 2), word(system, 3));
            let y2r = state(system);
            y2r.input_format = first & 0xFF;
            y2r.output_format = (first >> 8) & 0xFF;
            y2r.rotation = (first >> 16) & 0xFF;
            y2r.tiled = (first >> 24) & 0xFF != 0;
            y2r.line_width = second & 0xFFFF;
            y2r.lines = second >> 16;
            y2r.standard = third & 0xFF;
            y2r.coefficients = STANDARD[(third & 3) as usize];
            y2r.alpha = third >> 16;
            reply(system, &[]);
        }
        // DriverInitialize / DriverFinalize
        0x002B | 0x002C => {
            let event = state(system).event;
            system.services.y2r = Y2rState { event, ..Y2rState::default() };
            reply(system, &[]);
        }
        // GetPackageParameter
        0x002D => {
            let y2r = state(system);
            let first = y2r.input_format | y2r.output_format << 8 | y2r.rotation << 16 | (y2r.tiled as u32) << 24;
            let second = y2r.line_width | y2r.lines << 16;
            let third = y2r.standard | y2r.alpha << 16;
            reply(system, &[first, second, third]);
        }
        _ => return false,
    }
    true
}

fn state(system: &mut System) -> &mut Y2rState {
    &mut system.services.y2r
}

/// the end event, made the first time a title asks for it and held on to.
fn event(system: &mut System) -> ObjectId {
    if let Some(object) = system.services.y2r.event {
        return object;
    }
    let object = system
        .kernel
        .objects
        .insert(crate::kernel::object::KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "y2r:u end")));
    system.kernel.objects.add_ref(object);
    system.services.y2r.event = Some(object);
    object
}

/// count bytes of a buffer into out, a transfer's worth at a time with the
/// gaps skipped, keeping one byte of every width. every transfer is read
/// whole, the last one too.
fn gather(system: &mut System, from: Buffer, count: usize, width: usize, out: &mut Vec<u8>, piece: &mut Vec<u8>) {
    let transfer = (from.transfer as usize).max(width);
    let step = transfer as u32 + from.gap as u32;
    let mut address = from.address;
    if width == 1 {
        // every byte kept, the transfers go straight in
        out.resize(count.div_ceil(transfer) * transfer, 0);
        for piece in out.chunks_exact_mut(transfer) {
            system.memory.read_bytes(address, piece);
            address = address.wrapping_add(step);
        }
        out.truncate(count);
    } else {
        out.clear();
        piece.resize(transfer, 0);
        while out.len() < count {
            system.memory.read_bytes(address, piece);
            out.extend(piece.iter().step_by(width).take(count - out.len()));
            address = address.wrapping_add(step);
        }
    }
}

/// writes bytes to a buffer, a transfer's worth at a time with the gaps
/// skipped.
fn scatter(system: &mut System, to: Buffer, bytes: &[u8]) {
    let transfer = (to.transfer as usize).max(1);
    let mut address = to.address;
    for piece in bytes.chunks(transfer) {
        system.memory.write_bytes(address, piece);
        address = address.wrapping_add(transfer as u32 + to.gap as u32);
    }
}

/// the coefficients of a conversion, as the hardware's arithmetic goes once
/// its shifts are put together: a channel is the term of the pixel's Y and
/// a term of its U and V added, shifted down eight and clamped to a byte,
/// bit for bit what the hardware gives. it shifts down three, adds the
/// channel's offset and shifts down five more, and ((a >> 3) + k) >> 5 is
/// (a + 8k) >> 8.
#[derive(Debug, Clone, Copy)]
struct Terms {
    /// the coefficients multiplying Y, V for red, V and U for green and U
    /// for blue, halfwords all, which the vector code does better with.
    luma: i16,
    red_v: i16,
    green_v: i16,
    green_u: i16,
    blue_u: i16,
    /// eight times the offset of each channel.
    red: i32,
    green: i32,
    blue: i32,
}

impl Terms {
    fn new(c: &[i32; 8]) -> Terms {
        Terms {
            luma: c[0] as i16,
            red_v: c[1] as i16,
            green_v: c[2] as i16,
            green_u: c[3] as i16,
            blue_u: c[4] as i16,
            red: (c[5] + 0x18) * 8,
            green: (c[6] + 0x18) * 8,
            blue: (c[7] + 0x18) * 8,
        }
    }

    #[inline(always)]
    fn luma(&self, y: u8) -> i32 {
        self.luma as i32 * y as i32
    }

    /// the red, green and blue terms of a U and V.
    #[inline(always)]
    fn chroma(&self, u: u8, v: u8) -> [i32; 3] {
        let (u, v) = (u as i32, v as i32);
        [
            self.red_v as i32 * v + self.red,
            self.green - self.green_v as i32 * v - self.green_u as i32 * u,
            self.blue_u as i32 * u + self.blue,
        ]
    }
}

/// a channel from the sum of its terms. the clamp goes through 16 bits,
/// where the vector unit has a minimum and a maximum, 32 bits having none
/// before SSE4.1.
#[inline(always)]
fn channel(sum: i32) -> u32 {
    ((sum >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16).clamp(0, 0xFF) as u32
}

/// how the input of a conversion is laid out.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// planes of Y, U and V, a line of U and V going with this many lines
    /// of Y.
    Planar { lines_per_chroma: usize },
    /// a pair of pixels in four bytes, their Y, U, Y and V in turn.
    Yuyv,
}

/// the chroma terms of a line's pixels, the two of a U and V getting the
/// same.
fn chroma_terms(terms: &Terms, u: &[u8], v: &[u8], chroma: &mut [Vec<i32>; 3]) {
    let [red, green, blue] = chroma;
    let pairs = u.len();
    let (v, red, green, blue) = (&v[..pairs], &mut red[..2 * pairs], &mut green[..2 * pairs], &mut blue[..2 * pairs]);
    for k in 0..pairs {
        let [r, g, b] = terms.chroma(u[k], v[k]);
        red[2 * k..2 * k + 2].fill(r);
        green[2 * k..2 * k + 2].fill(g);
        blue[2 * k..2 * k + 2].fill(b);
    }
}

/// a line's pixels from its Y and chroma terms, as pack puts red, green and
/// blue together.
#[inline(always)]
fn line_pixels(terms: &Terms, luma: &[u8], chroma: &[Vec<i32>; 3], pixels: &mut [u32], pack: impl Fn(u32, u32, u32) -> u32) {
    let width = pixels.len();
    let (luma, red, green, blue) = (&luma[..width], &chroma[0][..width], &chroma[1][..width], &chroma[2][..width]);
    for x in 0..width {
        let y = terms.luma(luma[x]);
        pixels[x] = pack(channel(y + red[x]), channel(y + green[x]), channel(y + blue[x]));
    }
}

/// writes two pixels side by side, BYTES of each.
#[inline(always)]
fn put<const BYTES: usize>(pair: &[u32; 2], out: &mut [u8]) {
    let both = pair[0] as u64 | (pair[1] as u64) << (8 * BYTES);
    out.copy_from_slice(&both.to_le_bytes()[..2 * BYTES]);
}

/// puts a strip's pixels in its part of the output, in rows or in 8x8 tiles
/// in the order their pixels go in a texture, two at a time, as the two of
/// an even and odd column are side by side in a tile too.
fn place<const BYTES: usize>(pixels: &[u32], width: usize, rows: usize, tiled: bool, out: &mut [u8]) {
    use zakuro_gpu::format::morton_interleave;
    if !tiled {
        for (pair, out) in pixels[..rows * width].as_chunks::<2>().0.iter().zip(out[..rows * width * BYTES].chunks_exact_mut(2 * BYTES)) {
            put::<BYTES>(pair, out);
        }
    } else if rows == 8 {
        for (tile, out) in out[..8 * width * BYTES].chunks_exact_mut(64 * BYTES).enumerate() {
            for y in 0..8 {
                let pairs = pixels[y * width + tile * 8..][..8].as_chunks::<2>().0;
                for (x, pair) in (0..8).step_by(2).zip(pairs) {
                    let at = morton_interleave(x, y as u32) as usize * BYTES;
                    put::<BYTES>(pair, &mut out[at..at + 2 * BYTES]);
                }
            }
        }
    } else {
        // a last strip of fewer than eight lines, its tiles cut short
        for y in 0..rows {
            let pairs = pixels[y * width..][..width].as_chunks::<2>().0;
            for (x, pair) in (0..width).step_by(2).zip(pairs) {
                let at = (x / 8 * 64 + morton_interleave(x as u32, y as u32) as usize) * BYTES;
                put::<BYTES>(pair, &mut out[at..at + 2 * BYTES]);
            }
        }
    }
}

/// converts what was gathered into the output, eight lines at a time, BYTES
/// a pixel, a line's pixels first and then their places.
fn convert_strips<const BYTES: usize>(
    scratch: &mut Scratch,
    layout: Layout,
    terms: &Terms,
    width: usize,
    lines: usize,
    tiled: bool,
    pack: impl Fn(u32, u32, u32) -> u32 + Copy,
) {
    let Scratch { planes: [luma, chroma_u, chroma_v], line, chroma, pixels, out, .. } = scratch;
    let half = width / 2;
    out.resize(width * lines * BYTES, 0);
    pixels.resize(width * 8, 0);
    for part in chroma.iter_mut() {
        part.resize(width, 0);
    }
    let mut shared = None;
    for strip in (0..lines).step_by(8) {
        let rows = (lines - strip).min(8);
        for y in 0..rows {
            let at = strip + y;
            let luma_line = match layout {
                Layout::Planar { lines_per_chroma } => {
                    // the terms of a chroma line once for the two lines of
                    // 4:2:0 that share it
                    let chroma_line = at / lines_per_chroma;
                    if shared != Some(chroma_line) {
                        let (u, v) = (&chroma_u[chroma_line * half..][..half], &chroma_v[chroma_line * half..][..half]);
                        chroma_terms(terms, u, v, chroma);
                        shared = Some(chroma_line);
                    }
                    &luma[at * width..][..width]
                }
                Layout::Yuyv => {
                    let [y_line, u_line, v_line] = line;
                    y_line.resize(width, 0);
                    u_line.resize(half, 0);
                    v_line.resize(half, 0);
                    for (k, &[first, u, second, v]) in luma[at * width * 2..][..width * 2].as_chunks::<4>().0.iter().enumerate() {
                        y_line[2 * k] = first;
                        u_line[k] = u;
                        y_line[2 * k + 1] = second;
                        v_line[k] = v;
                    }
                    chroma_terms(terms, u_line, v_line, chroma);
                    &y_line[..]
                }
            };
            line_pixels(terms, luma_line, chroma, &mut pixels[y * width..][..width], pack);
        }
        place::<BYTES>(pixels, width, rows, tiled, &mut out[strip * width * BYTES..]);
    }
}

/// runs the conversion the title set up.
fn convert(system: &mut System) {
    let state = &system.services.y2r;
    let (width, lines) = (state.line_width as usize, state.lines as usize);
    if width == 0 || lines == 0 || !width.is_multiple_of(8) {
        log::warn!("y2r: nothing to convert, {width}x{lines}");
        return;
    }
    if state.rotation != 0 {
        log::warn!("y2r: rotation {} is not done, the image comes out upright", state.rotation);
    }
    let (format, output_format, tiled, alpha) = (state.input_format, state.output_format, state.tiled, state.alpha as u8);
    let terms = Terms::new(&state.coefficients);
    let (y_buffer, u_buffer, v_buffer, yuyv_buffer, output) = (state.y, state.u, state.v, state.yuyv, state.output);
    let pixels = width * lines;
    let mut scratch = std::mem::take(&mut system.services.y2r.scratch);
    let Scratch { planes: [luma, chroma_u, chroma_v], piece, .. } = &mut scratch;
    // 16 bit samples keep their low byte
    let sample = if format == 2 || format == 3 { 2 } else { 1 };
    let layout = match format {
        // 4:2:2 and 4:2:0, eight or sixteen bits a sample
        0 | 2 => {
            gather(system, y_buffer, pixels, sample, luma, piece);
            gather(system, u_buffer, pixels / 2, sample, chroma_u, piece);
            gather(system, v_buffer, pixels / 2, sample, chroma_v, piece);
            Layout::Planar { lines_per_chroma: 1 }
        }
        1 | 3 => {
            gather(system, y_buffer, pixels, sample, luma, piece);
            gather(system, u_buffer, pixels / 4, sample, chroma_u, piece);
            gather(system, v_buffer, pixels / 4, sample, chroma_v, piece);
            Layout::Planar { lines_per_chroma: 2 }
        }
        _ => {
            gather(system, yuyv_buffer, pixels * 2, 1, luma, piece);
            Layout::Yuyv
        }
    };
    let bytes = match output_format {
        // 32 bit with alpha first in memory and then blue, green and red, 24
        // bit without the alpha, the way the GPU has them
        0 => {
            let alpha = alpha as u32;
            convert_strips::<4>(&mut scratch, layout, &terms, width, lines, tiled, |r, g, b| alpha | b << 8 | g << 16 | r << 24);
            4
        }
        1 => {
            convert_strips::<3>(&mut scratch, layout, &terms, width, lines, tiled, |r, g, b| b | g << 8 | r << 16);
            3
        }
        // 16 bit 5-5-5-1 and 5-6-5
        2 => {
            let alpha = (alpha >= 0x80) as u32;
            convert_strips::<2>(&mut scratch, layout, &terms, width, lines, tiled, |r, g, b| (r >> 3) << 11 | (g >> 3) << 6 | (b >> 3) << 1 | alpha);
            2
        }
        _ => {
            convert_strips::<2>(&mut scratch, layout, &terms, width, lines, tiled, |r, g, b| (r >> 3) << 11 | (g >> 2) << 5 | b >> 3);
            2
        }
    };
    let end = output.address.wrapping_add((pixels * bytes) as u32 / output.transfer.max(1) as u32 * (output.transfer as u32 + output.gap as u32));
    system.sync_gpu(output.address, end.wrapping_sub(output.address));
    scatter(system, output, &scratch.out);
    system.services.y2r.scratch = scratch;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryRegion, MemoryState, Permission};
    use crate::Config;
    use std::collections::BTreeMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// the conversion as it was before it went a line at a time, kept to
    /// check the new one against.
    mod oracle {
        use super::super::{Buffer, System};

        pub fn gather(system: &mut System, from: Buffer, count: usize, width: usize) -> Vec<u8> {
            let transfer = (from.transfer as usize).max(width);
            let mut out = Vec::with_capacity(count);
            let mut address = from.address;
            let mut piece = vec![0u8; transfer];
            while out.len() < count {
                system.memory.read_bytes(address, &mut piece);
                out.extend(piece.iter().step_by(width).take(count - out.len()));
                address = address.wrapping_add(transfer as u32 + from.gap as u32);
            }
            out
        }

        fn scatter(system: &mut System, to: Buffer, bytes: &[u8]) {
            let transfer = (to.transfer as usize).max(1);
            let mut address = to.address;
            for piece in bytes.chunks(transfer) {
                system.memory.write_bytes(address, piece);
                address = address.wrapping_add(transfer as u32 + to.gap as u32);
            }
        }

        pub fn rgb(c: &[i32; 8], y: i32, u: i32, v: i32) -> [u8; 3] {
            let luma = c[0] * y;
            let r = ((luma + c[1] * v) >> 3) + c[5] + 0x18;
            let g = ((luma - c[2] * v - c[3] * u) >> 3) + c[6] + 0x18;
            let b = ((luma + c[4] * u) >> 3) + c[7] + 0x18;
            [r, g, b].map(|channel| (channel >> 5).clamp(0, 0xFF) as u8)
        }

        pub fn convert(system: &mut System) {
            let state = &system.services.y2r;
            let (width, lines) = (state.line_width as usize, state.lines as usize);
            if width == 0 || lines == 0 || !width.is_multiple_of(8) {
                log::warn!("y2r: nothing to convert, {width}x{lines}");
                return;
            }
            if state.rotation != 0 {
                log::warn!("y2r: rotation {} is not done, the image comes out upright", state.rotation);
            }
            let (format, coefficients, tiled, alpha) = (state.input_format, state.coefficients, state.tiled, state.alpha as u8);
            let (y_buffer, u_buffer, v_buffer, yuyv_buffer, output) = (state.y, state.u, state.v, state.yuyv, state.output);
            let pixels = width * lines;
            // 16 bit samples keep their low byte
            let sample = if format == 2 || format == 3 { 2 } else { 1 };
            let (luma, chroma_u, chroma_v, interleaved) = match format {
                // 4:2:2 and 4:2:0, eight or sixteen bits a sample
                0 | 2 => (
                    gather(system, y_buffer, pixels, sample),
                    gather(system, u_buffer, pixels / 2, sample),
                    gather(system, v_buffer, pixels / 2, sample),
                    Vec::new(),
                ),
                1 | 3 => (
                    gather(system, y_buffer, pixels, sample),
                    gather(system, u_buffer, pixels / 4, sample),
                    gather(system, v_buffer, pixels / 4, sample),
                    Vec::new(),
                ),
                _ => (Vec::new(), Vec::new(), Vec::new(), gather(system, yuyv_buffer, pixels * 2, 1)),
            };
            let yuv = |x: usize, y: usize| -> (i32, i32, i32) {
                match format {
                    0 | 2 => {
                        let i = y * width + x;
                        (luma[i] as i32, chroma_u[i / 2] as i32, chroma_v[i / 2] as i32)
                    }
                    1 | 3 => {
                        let i = (y / 2) * (width / 2) + x / 2;
                        (luma[y * width + x] as i32, chroma_u[i] as i32, chroma_v[i] as i32)
                    }
                    _ => {
                        let pair = (y * width + x / 2 * 2) * 2;
                        (interleaved[(y * width + x) * 2] as i32, interleaved[pair + 1] as i32, interleaved[pair + 3] as i32)
                    }
                }
            };
            let color = match system.services.y2r.output_format {
                0 => zakuro_gpu::format::ColorFormat::Rgba8,
                1 => zakuro_gpu::format::ColorFormat::Rgb8,
                2 => zakuro_gpu::format::ColorFormat::Rgb5A1,
                _ => zakuro_gpu::format::ColorFormat::Rgb565,
            };
            let bytes = color.bytes_per_pixel();
            // eight lines at a time, in rows or in 8x8 tiles in the order their
            // pixels go in a texture
            let mut out = vec![0u8; pixels * bytes];
            let mut at = 0;
            for strip in (0..lines).step_by(8) {
                let rows = (lines - strip).min(8);
                let mut place = |x: usize, y: usize| {
                    let [r, g, b] = {
                        let (luma, u, v) = yuv(x, strip + y);
                        rgb(&coefficients, luma, u, v)
                    };
                    let index = if tiled {
                        (x / 8) * 64 + zakuro_gpu::format::morton_offset(x as u32 % 8, y as u32, 8, 1) as usize
                    } else {
                        y * width + x
                    };
                    color.encode([r, g, b, alpha], &mut out[at + index * bytes..at + (index + 1) * bytes]);
                };
                for y in 0..rows {
                    for x in 0..width {
                        place(x, y);
                    }
                }
                at += rows * width * bytes;
            }
            let end = output.address.wrapping_add((pixels * bytes) as u32 / output.transfer.max(1) as u32 * (output.transfer as u32 + output.gap as u32));
            system.sync_gpu(output.address, end.wrapping_sub(output.address));
            scatter(system, output, &out);
        }
    }

    /// one pixel the way a conversion works it out.
    fn rgb(c: &[i32; 8], y: u8, u: u8, v: u8) -> [u8; 3] {
        let terms = Terms::new(c);
        let luma = terms.luma(y);
        terms.chroma(u, v).map(|term| channel(luma + term) as u8)
    }

    /// black, white and grey come out the way BT.601 says, full range.
    #[test]
    fn converts_with_the_hardware_arithmetic() {
        let c = &STANDARD[0];
        // the coefficients round, a channel can be one off
        assert!(rgb(c, 0, 128, 128).iter().all(|&v| v <= 1));
        assert!(rgb(c, 255, 128, 128).iter().all(|&v| v >= 254));
        let grey = rgb(c, 128, 128, 128);
        assert!(grey.iter().all(|&v| (127..=129).contains(&v)), "{grey:?}");
        // pure red in YUV
        let red = rgb(c, 76, 85, 255);
        assert!(red[0] > 240 && red[1] < 15 && red[2] < 15, "{red:?}");
    }

    /// numbers for the random cases, the same every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }

        fn coefficients(&mut self) -> [i32; 8] {
            match self.below(3) {
                0 => std::array::from_fn(|_| self.next() as u16 as i16 as i32),
                // at the ends of a halfword, where sums leave 16 bits
                1 => std::array::from_fn(|_| [i16::MIN, i16::MAX, -1, 0, 1][self.below(5) as usize] as i32),
                _ => STANDARD[self.below(4) as usize],
            }
        }
    }

    /// the folded arithmetic gives what the hardware's does for every Y, U
    /// and V, under the standard coefficients, random ones and ones whose
    /// sums go past what 16 bits hold.
    #[test]
    fn folds_the_arithmetic_exactly() {
        let mut random = Random(1);
        let mut sets = STANDARD.to_vec();
        sets.extend((0..12).map(|_| random.coefficients()));
        sets.extend([[i16::MAX as i32; 8], [i16::MIN as i32; 8]]);
        for c in &sets {
            for y in 0..=255u8 {
                for u in (0..=255u8).step_by(5) {
                    for v in 0..=255u8 {
                        let expected = oracle::rgb(c, y as i32, u as i32, v as i32);
                        assert_eq!(rgb(c, y, u, v), expected, "{c:?} at {y} {u} {v}");
                    }
                }
            }
        }
    }

    /// a strip's pixels go where the GPU looks for them in a texture, a whole
    /// strip and a last one of four lines alike, and in rows otherwise, at
    /// every pixel size.
    #[test]
    fn places_pixels_where_textures_have_them() {
        fn check<const BYTES: usize>(width: usize, rows: usize, alignments: &[bool]) {
            let mask = (1u64 << (8 * BYTES)) - 1;
            let pixels: Vec<u32> = (0..width * 8).map(|i| ((i as u64 * 0x0105_0309) & mask) as u32).collect();
            for &tiled in alignments {
                let mut out = vec![0u8; width * rows * BYTES];
                place::<BYTES>(&pixels, width, rows, tiled, &mut out);
                for y in 0..rows {
                    for x in 0..width {
                        let index = if tiled {
                            x / 8 * 64 + zakuro_gpu::format::morton_offset(x as u32 % 8, y as u32, 8, 1) as usize
                        } else {
                            y * width + x
                        };
                        let pixel = pixels[y * width + x].to_le_bytes();
                        assert_eq!(out[index * BYTES..][..BYTES], pixel[..BYTES], "{BYTES} bytes, {width}x{rows}, tiled {tiled}, at {x} {y}");
                    }
                }
            }
        }
        // tiles cut short only fit as a strip of four lines one tile wide
        for (width, rows, alignments) in [(8, 8, &[false, true][..]), (40, 8, &[false, true]), (8, 4, &[false, true]), (16, 3, &[false])] {
            check::<4>(width, rows, alignments);
            check::<3>(width, rows, alignments);
            check::<2>(width, rows, alignments);
        }
    }

    /// guest memory the cases convert in, the pages around it unmapped.
    const BASE: u32 = 0x0800_0000;
    const SIZE: u32 = 0x10_0000;

    fn system() -> System {
        let mut system = System::new(Config::default());
        let block = system.memory.phys.allocate(MemoryRegion::Application, SIZE).unwrap();
        system.memory.map(BASE, block.addr, SIZE, Permission::RW, MemoryState::Private);
        system
    }

    /// noise over all of the memory.
    fn fill(system: &mut System, random: &mut Random) {
        let noise: Vec<u8> = (0..SIZE / 4).flat_map(|_| random.next().to_le_bytes()).collect();
        system.memory.write_bytes(BASE, &noise);
    }

    fn memory(system: &mut System) -> Vec<u8> {
        let mut bytes = vec![0; SIZE as usize];
        system.memory.read_bytes(BASE, &mut bytes);
        bytes
    }

    /// how many times each unmapped page was touched so far.
    fn faults(system: &System) -> BTreeMap<u32, u32> {
        system.memory.fault_summary().into_iter().collect()
    }

    /// how many more times each unmapped page was touched by after than by
    /// before.
    fn touched(before: &BTreeMap<u32, u32>, after: &BTreeMap<u32, u32>) -> BTreeMap<u32, u32> {
        after.iter().map(|(&page, &count)| (page, count - before.get(&page).copied().unwrap_or(0))).filter(|&(_, count)| count > 0).collect()
    }

    /// a buffer in the mapped memory or running off it, with transfers and
    /// gaps of every kind.
    fn buffer(random: &mut Random, width: usize) -> Buffer {
        let transfer = match random.below(6) {
            0 => random.below(4),
            1 => width as u32,
            2 => width as u32 * 8 * (1 + random.below(4)),
            3 => random.below(0x10000),
            _ => 1 + random.below(600),
        } as u16;
        let gap = match random.below(8) {
            0..=3 => 0,
            4 | 5 => random.below(16),
            6 => random.below(600),
            _ => random.below(0x10000),
        } as u16;
        let address = match random.below(10) {
            // near either end of the memory, the rest unmapped
            0 => BASE + SIZE - random.below(0x2000),
            1 => BASE - random.below(0x2000),
            _ => BASE + random.below(SIZE / 2),
        };
        Buffer { address, transfer, gap }
    }

    /// converts with the old code and then the new from the same memory and
    /// settings, and checks the two leave memory the same and touch the same
    /// unmapped pages as often.
    fn compare(system: &mut System, case: &str) {
        let before = memory(system);
        let first = faults(system);
        oracle::convert(system);
        let expected = memory(system);
        let second = faults(system);
        system.memory.write_bytes(BASE, &before);
        convert(system);
        let actual = memory(system);
        let third = faults(system);
        if let Some(at) = expected.iter().zip(&actual).position(|(a, b)| a != b) {
            panic!("{case}: memory differs from 0x{:08X}", BASE + at as u32);
        }
        assert_eq!(touched(&first, &second), touched(&second, &third), "{case}: unmapped pages");
    }

    /// conversions of every input and output format, rotation, alignment,
    /// set of coefficients, alpha and size come out as they did before,
    /// read from and written to buffers cut up every way.
    #[test]
    fn converts_as_before() {
        let mut system = system();
        let mut random = Random(2);
        for case in 0..400 {
            fill(&mut system, &mut random);
            let input_format = if random.below(8) == 0 { random.below(0x100) } else { random.below(5) };
            let output_format = if random.below(8) == 0 { random.below(0x100) } else { random.below(4) };
            let tiled = random.below(2) == 0;
            let width = match random.below(10) {
                0 => random.below(100) as usize,
                _ => 8 * (1 + random.below(12)) as usize,
            };
            let mut lines = 1 + random.below(40) as usize;
            // the shapes the old code ran off its buffers on are checked on
            // their own
            if (input_format == 1 || input_format == 3) && lines % 2 == 1 {
                lines += 1;
            }
            if tiled && !lines.is_multiple_of(8) && !(width == 8 && lines % 8 == 4) {
                lines = lines.next_multiple_of(8);
            }
            let y2r = &mut system.services.y2r;
            y2r.input_format = input_format;
            y2r.output_format = output_format;
            y2r.rotation = random.below(4);
            y2r.tiled = tiled;
            y2r.line_width = width as u32;
            y2r.lines = lines as u32;
            y2r.coefficients = random.coefficients();
            // the 5-5-5-1 bit turns on at 0x80 of the low byte
            y2r.alpha = [0x7F, 0x80, 0x17F, 0x180, random.below(0x10000)][random.below(5) as usize];
            for target in [&mut y2r.y, &mut y2r.u, &mut y2r.v, &mut y2r.yuyv, &mut y2r.output] {
                *target = buffer(&mut random, width);
            }
            let settings = format!("case {case}, {input_format} to {output_format}, {width}x{lines}, tiled {tiled}");
            compare(&mut system, &settings);
        }
    }

    /// the frames two titles play their movies in, MH4U's lines one after
    /// another and Kirby's with gaps, come out as they did before in every
    /// output format.
    #[test]
    fn converts_movie_frames_as_before() {
        let mut system = system();
        let mut random = Random(4);
        let frames = [
            (512, [(512, 0), (256, 256), (256, 256)], 12288, 0),
            (400, [(400, 112), (200, 312), (200, 312)], 9600, 2688),
        ];
        for (width, planes, transfer, gap) in frames {
            for output_format in 0..4 {
                for tiled in [true, false] {
                    fill(&mut system, &mut random);
                    let y2r = &mut system.services.y2r;
                    y2r.input_format = 1;
                    y2r.output_format = output_format;
                    y2r.tiled = tiled;
                    y2r.line_width = width;
                    y2r.lines = 240;
                    y2r.coefficients = STANDARD[output_format as usize];
                    y2r.alpha = 0xFF;
                    for (target, (at, (transfer, gap))) in [&mut y2r.y, &mut y2r.u, &mut y2r.v].into_iter().zip([0, 0x2_0000, 0x4_0000].into_iter().zip(planes)) {
                        *target = Buffer { address: BASE + at, transfer, gap };
                    }
                    let bytes = [4, 3, 2, 2][output_format as usize];
                    y2r.output = Buffer { address: BASE + 0x8_0000, transfer: transfer / 3 * bytes, gap: gap / 3 * bytes };
                    compare(&mut system, &format!("{width} wide to {output_format}, tiled {tiled}"));
                }
            }
        }
    }

    /// eight and 16 bit samples are gathered as they were, a transfer at a
    /// time whatever its size, the reads the same down to the unmapped pages
    /// they touch.
    #[test]
    fn gathers_as_before() {
        let mut system = system();
        let mut random = Random(5);
        fill(&mut system, &mut random);
        let (mut out, mut piece) = (Vec::new(), Vec::new());
        for case in 0..600 {
            let width = 1 + random.below(2) as usize;
            let count = random.below(3000) as usize;
            let line = 1 + random.below(64) as usize;
            let from = buffer(&mut random, line);
            let first = faults(&system);
            let expected = oracle::gather(&mut system, from, count, width);
            let second = faults(&system);
            gather(&mut system, from, count, width, &mut out, &mut piece);
            let third = faults(&system);
            assert_eq!(out, expected, "case {case}, {count} of {width} from {from:?}");
            assert_eq!(touched(&first, &second), touched(&second, &third), "case {case}, unmapped pages");
        }
    }

    /// the shapes the old code ran off its buffers on, 4:2:0 with an odd
    /// number of lines and tiles cut short but for a strip of four lines
    /// one tile wide, still stop a conversion, and the ones it took still
    /// convert the same.
    #[test]
    fn stops_where_it_stopped() {
        let mut system = system();
        let mut random = Random(6);
        let mut stopped = 0;
        for (input_format, tiled, width, lines) in [
            (1, false, 16, 3),
            (3, false, 8, 1),
            (1, true, 8, 13),
            (0, true, 16, 4),
            (0, true, 8, 3),
            (0, true, 8, 12),
            (4, true, 8, 4),
            (2, true, 24, 7),
            (4, true, 8, 6),
            (1, true, 8, 20),
        ] {
            fill(&mut system, &mut random);
            let y2r = &mut system.services.y2r;
            y2r.input_format = input_format;
            y2r.output_format = random.below(4);
            y2r.tiled = tiled;
            y2r.line_width = width;
            y2r.lines = lines;
            y2r.coefficients = STANDARD[0];
            for (index, target) in [&mut y2r.y, &mut y2r.u, &mut y2r.v, &mut y2r.yuyv, &mut y2r.output].into_iter().enumerate() {
                *target = Buffer { address: BASE + index as u32 * 0x1_0000, transfer: width as u16, gap: 0 };
            }
            let case = format!("{input_format}, {width}x{lines}, tiled {tiled}");
            let before = memory(&mut system);
            let old = catch_unwind(AssertUnwindSafe(|| oracle::convert(&mut system))).is_err();
            system.memory.write_bytes(BASE, &before);
            let new = catch_unwind(AssertUnwindSafe(|| convert(&mut system))).is_err();
            assert_eq!(new, old, "{case}: stopped");
            if old {
                stopped += 1;
            } else {
                system.memory.write_bytes(BASE, &before);
                compare(&mut system, &case);
            }
        }
        assert_eq!(stopped, 7);
    }
}
