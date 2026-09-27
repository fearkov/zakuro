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

/// count bytes of a buffer, a transfer's worth at a time with the gaps
/// skipped, keeping one byte of every width.
fn gather(system: &mut System, from: Buffer, count: usize, width: usize) -> Vec<u8> {
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

/// one pixel's red, green and blue from its Y, U and V, bit for bit what the
/// hardware gives.
fn rgb(c: &[i32; 8], y: i32, u: i32, v: i32) -> [u8; 3] {
    let luma = c[0] * y;
    let r = ((luma + c[1] * v) >> 3) + c[5] + 0x18;
    let g = ((luma - c[2] * v - c[3] * u) >> 3) + c[6] + 0x18;
    let b = ((luma + c[4] * u) >> 3) + c[7] + 0x18;
    [r, g, b].map(|channel| (channel >> 5).clamp(0, 0xFF) as u8)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
