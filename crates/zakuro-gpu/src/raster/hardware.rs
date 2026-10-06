//! filling triangles on the host's GPU through Vulkan. the GPU runs the
//! title's vertex shader, clips and culls, then does what costs, the
//! pixels, texturing, lighting, the combiners, the tests and blending. a
//! draw with a geometry shader has the CPU work out its triangles the way
//! the software path does instead.
//!
//! guest memory stays where images live between command lists. a buffer a
//! list draws into goes up to the GPU the first time the list needs it,
//! unless it is still as the GPU left it, and comes back down only when
//! something else is about to read it, a texture, a fill or the CPU.
//!
//! a display transfer out of such a buffer runs on the GPU as well, then its
//! batch goes to the GPU without waiting for it, along with a copy of the
//! output for the host to read later. recording goes on in a second command
//! buffer meanwhile, so the CPU rarely waits for the GPU.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::sync::{mpsc, Arc, Condvar, Mutex, PoisonError};
use std::thread;

use ash::vk;

use super::{BoundTexture, DepthMap, DrawnTexture, Screen, Wrap, TEXTURE_UNIT_BASES};
use crate::blend::LogicOp;
use crate::fog;
use crate::proctex;
use crate::{Picture, ScreenRef};
use crate::format::{morton_offset, ColorFormat};
use crate::lighting::{Lighting, Tables};
use crate::registers::*;
use crate::device::SharedDevice;
use crate::shader::glsl::{self, SEMANTICS};
use crate::shader::{Program, ShaderUnit, Vec4, DESCRIPTOR_SIZE, INPUT_REGISTERS, PROGRAM_SIZE};
use crate::GpuMemory;
use hasher::{QuickMap, QuickSet};

mod hasher;

const SHADE_SPIRV: &[u8] = include_bytes!("../../shaders/shade.vert.spv");
const VERTEX_SPIRV: &[u8] = include_bytes!("../../shaders/raster.vert.spv");
const FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/raster.frag.spv");
/// raster.frag built with WRITES_DEPTH, for the draws whose depth the
/// rasterizer can't work out.
const FRAGMENT_DEPTH_SPIRV: &[u8] = include_bytes!("../../shaders/raster_depth.frag.spv");
const TRANSFER_SPIRV: &[u8] = include_bytes!("../../shaders/transfer.comp.spv");
const DEPTH_SPIRV: &[u8] = include_bytes!("../../shaders/depth.comp.spv");
const UPRIGHT_SPIRV: &[u8] = include_bytes!("../../shaders/upright.comp.spv");
/// upright.comp built with TO_IMAGE, for a presenter sharing the device.
const UPRIGHT_IMAGE_SPIRV: &[u8] = include_bytes!("../../shaders/upright_image.comp.spv");

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D24_UNORM_S8_UINT;

/// where a batch keeps its vertices, uniforms and uploads.
const RING_SIZE: u64 = 64 << 20;
/// batches the GPU may be running while the next one is recorded, so the
/// emulation goes on meanwhile.
const IN_FLIGHT: usize = 3;
/// a batch past this much is run before the next draw, so a draw always
/// finds room.
const RING_FLUSH: u64 = RING_SIZE / 2;

/// a vertex is six vectors, position, color, texture coordinates 0 and 1,
/// coordinate 2 with the mapped depth, the quaternion and the view vector.
const VERTEX_SIZE: usize = 24 * 4;
/// the words of a draw's uniform block, the combiners, the texture units
/// and the flags, then the lighting.
/// the procedural texture's registers, at the end of the uniforms.
const PROCTEX_WORDS: usize = 8;
/// the fog's color, and room.
const FOG_WORDS: usize = 4;
const UNIFORM_WORDS: usize = 12 * 4 + 4 + 3 * 4 + 4 + 328 + PROCTEX_WORDS + FOG_WORDS;
const UNIFORM_SIZE: u64 = (UNIFORM_WORDS * 4) as u64;
/// 24 lighting tables of 256 entries, each a value and a step.
/// the lighting tables, then the procedural texture's, its noise, color
/// map and alpha map as values and steps and its colors and their steps as
/// two pairs each, then the fog's, a value and a step an entry.
const TABLES_SIZE: u64 = (24 * 256 + 3 * proctex::MAP_ENTRIES + 4 * proctex::COLOR_ENTRIES + fog::ENTRIES) as u64 * 8;
/// a shader program and its operand descriptors.
const PROGRAM_BYTES: u64 = ((PROGRAM_SIZE + DESCRIPTOR_SIZE) * 4) as u64;
/// the words of what a vertex shader reads besides its program, the float
/// uniforms, the integer ones, the bools and the entry point, the
/// semantics, the depth map, the viewport, and where each input register
/// finds its attribute and what it reads without one.
const SHADING_WORDS: usize = 96 * 4 + 4 * 4 + 4 + 24 + 4 + 4 + 16 * 4 + 16 * 4;
const SHADING_SIZE: u64 = (SHADING_WORDS * 4) as u64;
/// a semantic no output register carries, which takes its default.
pub(super) const MISSING: u32 = u32::MAX;
/// a semantic in an output attribute past the enabled registers, zero.
pub(super) const ZERO: u32 = u32::MAX - 1;

/// where each combiner stage's registers start.
const STAGE_REGISTERS: [usize; 6] = [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8];
const REG_UPDATE_BUFFER: usize = 0x0E0;
const REG_BUFFER_COLOR: usize = 0x0FD;
const REG_BLEND_COLOR: usize = 0x103;

/// the PICA's comparisons in its own order.
const COMPARES: [vk::CompareOp; 8] = [
    vk::CompareOp::NEVER,
    vk::CompareOp::ALWAYS,
    vk::CompareOp::EQUAL,
    vk::CompareOp::NOT_EQUAL,
    vk::CompareOp::LESS,
    vk::CompareOp::LESS_OR_EQUAL,
    vk::CompareOp::GREATER,
    vk::CompareOp::GREATER_OR_EQUAL,
];

/// what a draw needs from the rasterizer.
pub(super) struct Draw<'a> {
    pub(super) registers: &'a [u32],
    pub(super) target: u32,
    pub(super) format: ColorFormat,
    pub(super) width: u32,
    pub(super) height: u32,
    /// the pixels the draw may touch, left, bottom, right and top in window
    /// coordinates, y up.
    pub(super) scissor: [i32; 4],
    /// the depth and stencil buffer and its bytes per sample, when the draw
    /// uses one.
    pub(super) depth: Option<(u32, u32)>,
    pub(super) depth_map: DepthMap,
    pub(super) geometry: Geometry<'a>,
    pub(super) textures: &'a [Option<BoundTexture>; 3],
    pub(super) lighting: Option<&'a Lighting>,
    pub(super) tables: &'a Tables,
    /// whether the procedural texture is on, and its tables.
    pub(super) proctex: bool,
    pub(super) proctex_tables: &'a proctex::Tables,
    /// whether the fog is on, and its table.
    pub(super) fog: bool,
    pub(super) fog_table: &'a fog::Table,
}

/// the triangles of a draw.
#[derive(Clone, Copy)]
pub(super) enum Geometry<'a> {
    /// shaded, clipped and culled on the CPU, placed vertices and the
    /// triangles, three indices each, that share them.
    Placed { vertices: &'a [Screen], indices: &'a [u32] },
    /// for the GPU to shade.
    Shaded(&'a Shading<'a>),
}

/// vertices the GPU runs the vertex shader over.
pub(super) struct Shading<'a> {
    pub(super) unit: &'a ShaderUnit,
    pub(super) inputs: Inputs<'a>,
    /// three inputs a triangle.
    pub(super) indices: &'a [u32],
    /// where each varying is in the output registers.
    pub(super) semantics: [u32; 24],
    /// left, bottom, width and height.
    pub(super) viewport: (f32, f32, f32, f32),
    /// the face culling register.
    pub(super) cull: u32,
}

/// a draw's input registers, as the vertex shader gets them.
#[derive(Clone, Copy)]
pub(super) enum Inputs<'a> {
    /// each vertex's registers, worked out on the CPU.
    Decoded(&'a [[Vec4; INPUT_REGISTERS]]),
    /// the vertex arrays as guest memory holds them, the shader working
    /// out each vertex's registers itself.
    Raw(&'a RawInputs),
}

/// a draw's vertex arrays for the GPU to decode.
#[derive(Default)]
pub(super) struct RawInputs {
    /// each array's bytes from the first vertex the draw uses to the last,
    /// as an address and a length in one piece of host memory.
    pub(super) arrays: Vec<(u32, u32)>,
    /// for each input register, the attribute an array gives it.
    pub(super) fields: [Option<RawField>; INPUT_REGISTERS],
    /// what each register reads where no attribute gives a component.
    pub(super) defaults: [Vec4; INPUT_REGISTERS],
}

/// where an attribute is in its array, and what it is.
#[derive(Clone, Copy)]
pub(super) struct RawField {
    pub(super) array: usize,
    /// from one vertex to the next, and from a vertex's start, in bytes.
    pub(super) stride: u32,
    pub(super) offset: u32,
    /// signed byte, byte, signed short or float, and how many components.
    pub(super) ty: u32,
    pub(super) count: u32,
}

/// a block of words a draw stages, the uniforms or what else the vertex
/// shader reads, and the last one of its kind the batch holds.
#[derive(Default)]
struct Block {
    /// the words being gathered.
    words: Vec<u32>,
    /// the words staged last, and where, until the batch is handed over.
    last: Vec<u32>,
    at: Option<u64>,
}

impl Block {
    /// where the batch holds the words gathered, when they are the words
    /// the last block of the kind staged.
    fn staged(&self) -> Option<u64> {
        self.at.filter(|_| self.words == self.last)
    }

    /// the words gathered went into the batch at offset.
    fn staged_at(&mut self, offset: u64) {
        std::mem::swap(&mut self.words, &mut self.last);
        self.at = Some(offset);
    }
}

fn vk_error(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |error| format!("could not {what}, {error}")
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
    /// the host's view needs invalidating before it reads what the GPU wrote.
    incoherent: bool,
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Color(ColorFormat),
    /// bytes per sample, 2 for D16, 3 for D24 and 4 for D24S8.
    Depth(u32),
}

impl Kind {
    fn bytes(self) -> u32 {
        match self {
            Kind::Color(format) => format.bytes_per_pixel() as u32,
            Kind::Depth(bytes) => bytes,
        }
    }
}

/// a color or depth buffer of the guest's, kept on the GPU with its rows
/// bottom first, the other way from memory, see draw.
struct Surface {
    image: Image,
    addr: u32,
    width: u32,
    height: u32,
    kind: Kind,
    /// tiled the way the PICA draws, or in plain rows, as a display
    /// transfer can leave its output.
    tiled: bool,
    /// the guest's bytes the last time the image matched them.
    shadow: Vec<u8>,
    /// the pixel all of those are, when a fill left them so or memory held
    /// nothing else, standing in for them. a frame's fills would copy
    /// megabytes into shadows, and memory compares faster with a pixel.
    pixel: Option<[u8; 4]>,
    /// the image is known to match guest memory in this batch.
    checked: bool,
    /// the rows drawn into since guest memory last got the image, rows of
    /// memory from the buffer's start, whole rows of tiles.
    dirty: Option<(u32, u32)>,
    /// the rows of those whose reads the CPU was told to ask for first.
    guarded: Option<(u32, u32)>,
    /// and those its writes have to wait for, see guard_writes.
    write_guarded: Option<(u32, u32)>,
    /// rows a fill cleared while others it drew had not come down, which
    /// another surface over the same memory may have drawn over since.
    cleared: Option<(u32, u32)>,
    capture: Option<Capture>,
    /// when drawing scaled, an image at the console's resolution the
    /// surface goes through on its way to and from guest memory, and a
    /// copy of the scaled image for showing it.
    native: Option<Image>,
    screen: Option<Capture>,
    /// the picture screen would copy out, kept on the GPU instead when the
    /// presenter shares the device.
    upright: Option<Upright>,
    /// counts the changes to the image, for the textures copied from it.
    generation: u64,
}

impl Surface {
    fn size(&self) -> u32 {
        self.width * self.height * self.kind.bytes()
    }

    fn overlaps(&self, addr: u32, len: u32) -> bool {
        addr < self.addr + self.size() && self.addr < addr + len
    }

    /// the bytes a row of the buffer takes.
    fn row_bytes(&self) -> u32 {
        self.width * self.kind.bytes()
    }

    /// whether the shadow holds bytes over a range of the buffer's memory,
    /// one starting at a pixel. the shadow holds all of the buffer or none.
    fn shadows(&self, range: std::ops::Range<usize>, bytes: &[u8]) -> bool {
        match self.pixel {
            Some(pixel) => {
                debug_assert!(range.start.is_multiple_of(self.kind.bytes() as usize));
                range.end <= self.size() as usize
                    && bytes.len() == range.len()
                    && crate::pattern::holds(bytes, &pixel[..self.kind.bytes() as usize])
            }
            None => self.shadow.get(range) == Some(bytes),
        }
    }

    /// has the shadow hold the buffer's bytes as memory does, through the
    /// pixel they all are when they are, as memory under a buffer the GPU
    /// draws often is.
    fn keep(&mut self, bytes: &[u8]) {
        let bpp = self.kind.bytes() as usize;
        let size = self.size() as usize;
        match bytes.get(..bpp).filter(|pixel| bytes.len() == size && crate::pattern::holds(bytes, pixel)) {
            Some(pixel) => self.keep_pixel(pixel),
            None => {
                // copied, so each shadow keeps a size of its own
                self.pixel = None;
                self.shadow.clear();
                self.shadow.extend_from_slice(bytes);
            }
        }
    }

    /// has the shadow hold the pixel all over the buffer.
    fn keep_pixel(&mut self, pixel: &[u8]) {
        let mut kept = [0; 4];
        kept[..pixel.len()].copy_from_slice(pixel);
        self.pixel = Some(kept);
    }

    /// has the shadow hold the pixel over a range of the buffer's memory,
    /// one starting at a pixel, when it holds the buffer at all.
    fn keep_rows(&mut self, range: std::ops::Range<usize>, pixel: &[u8]) {
        // a shadow all of the same pixel holds them already
        if self.pixel.is_none_or(|kept| kept[..pixel.len()] != *pixel) {
            if let Some(shadow) = self.shadow().get_mut(range) {
                crate::pattern::fill(shadow, pixel);
            }
        }
    }

    /// the shadow's bytes, written out where a pixel stood in for them.
    fn shadow(&mut self) -> &mut Vec<u8> {
        if let Some(pixel) = self.pixel.take() {
            self.shadow.resize(self.size() as usize, 0);
            crate::pattern::fill(&mut self.shadow, &pixel[..self.kind.bytes() as usize]);
        }
        &mut self.shadow
    }

    /// whether guest memory over a range is behind the image, where the GPU
    /// drew. titles pack buffers tightly, one's unused rows can be
    /// another's, and only rows really drawn matter.
    fn dirty_overlaps(&self, addr: u32, len: u32) -> bool {
        self.dirty.is_some_and(|(start, end)| {
            let (from, to) = (self.addr + start * self.row_bytes(), self.addr + end * self.row_bytes());
            addr < to && from < addr + len
        })
    }

    /// the GPU changed rows of the image, rows of memory from the start.
    fn drew(&mut self, (start, end): (u32, u32)) {
        let rows = ((start / 8 * 8).min(self.height), end.div_ceil(8).saturating_mul(8).min(self.height));
        self.dirty = Some(match self.dirty {
            Some((from, to)) => (from.min(rows.0), to.max(rows.1)),
            None => rows,
        });
        self.replaced();
    }

    /// the GPU changed the whole image.
    fn changed(&mut self) {
        self.drew((0, self.height));
    }

    /// has the CPU's reads of the rows drawn ask for them first, until
    /// guest memory gets them. draw does it for depth buffers alone.
    fn guard<M: GpuMemory>(&mut self, memory: &mut M) {
        let row = self.row_bytes();
        if let Some((addr, len)) = widen(&mut self.guarded, self.dirty, self.addr, row) {
            memory.guard(addr, len);
        }
    }

    /// has the rows drawn reach guest memory before the CPU writes over
    /// them, as on the console, where the GPU put its pixels there at once.
    /// a title can put something else where a buffer it is done with was,
    /// which the drawing written back later would break.
    fn guard_writes<M: GpuMemory>(&mut self, memory: &mut M) {
        let row = self.row_bytes();
        if let Some((addr, len)) = widen(&mut self.write_guarded, self.dirty, self.addr, row) {
            memory.guard_writes(addr, len);
        }
    }

    /// the image holds something new, the copies of it are old.
    fn replaced(&mut self) {
        self.generation += 1;
        for capture in [&mut self.capture, &mut self.screen].into_iter().flatten() {
            capture.current = false;
        }
        if let Some(upright) = &mut self.upright {
            upright.current = false;
        }
    }

    /// what a capture holds, when it still has the image.
    fn captured(&self) -> Option<&Capture> {
        self.capture.as_ref().filter(|capture| self.dirty.is_some() && capture.current)
    }
}

/// where the pipelines compiled before are kept, the system's place for
/// caches. the tests keep none, theirs are no use to anyone playing.
fn pipeline_cache_path() -> Option<std::path::PathBuf> {
    if cfg!(test) {
        return None;
    }
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(std::path::PathBuf::from);
    let dir = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library/Caches"))
    } else {
        var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|home| home.join(".cache")))
    };
    dir.map(|dir| dir.join("zakuro").join("pipelines.bin"))
}

/// keeps the pipelines compiled so far for the next run, by way of a file
/// of its own, so a run killed halfway leaves the last one whole.
fn save_pipeline_cache(device: &ash::Device, cache: vk::PipelineCache) {
    let Some(path) = pipeline_cache_path() else {
        return;
    };
    // SAFETY: the cache is the device's, and taking its data needs no one
    // else to keep off it
    let data = match unsafe { device.get_pipeline_cache_data(cache) } {
        Ok(data) => data,
        Err(error) => {
            log::warn!("could not read the pipeline cache, {error}");
            return;
        }
    };
    let partial = path.with_extension(format!("{}.partial", std::process::id()));
    let size = data.len();
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&partial, data))
        .and_then(|()| std::fs::rename(&partial, &path));
    match written {
        Ok(()) => log::debug!(target: "zakuro_gpu::pipelines", "kept {} KB of pipelines in {}", size / 1024, path.display()),
        Err(error) => {
            log::warn!("could not save the pipeline cache to {}, {error}", path.display());
            let _ = std::fs::remove_file(&partial);
        }
    }
}

/// how long the compiler thread waits between keeping what it compiled, a
/// run that does not end well losing no more than that.
const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// how much a surface holds the latest of some of its rows of memory, of
/// those over the same rows, what the GPU drew there and guest memory lacks
/// first, then the taller, which titles draw a shorter buffer into the start
/// of. drawn in other rows, it may have older ones there.
fn newest(surface: &Surface, (start, end): (u32, u32)) -> (bool, u32) {
    (surface.dirty.is_some_and(|(from, to)| from < end && start < to), surface.height)
}

/// the rows of memory of a tiled surface a fill covers, when it covers some
/// of its rows of tiles whole but not all of them.
fn rows_filled(surface: &Surface, addr: u32, len: u32) -> Option<(u32, u32)> {
    let row = surface.row_bytes() as u64;
    let start = (addr as u64).max(surface.addr as u64) - surface.addr as u64;
    let end = (addr as u64 + len as u64).min(surface.addr as u64 + surface.size() as u64);
    let end = end.checked_sub(surface.addr as u64)?;
    let whole = |offset: u64| offset.is_multiple_of(row * 8);
    let partial = start > 0 || end < surface.size() as u64;
    (surface.tiled && end > start && partial && whole(start) && whole(end)).then(|| ((start / row) as u32, (end / row) as u32))
}

/// the rows of a span left once other rows are taken out of it, which can
/// only shorten it at either end, rows in its middle leave all of it.
fn without((start, end): (u32, u32), (from, to): (u32, u32)) -> Option<(u32, u32)> {
    if from <= start && end <= to {
        None
    } else if from <= start && start < to {
        Some((to, end))
    } else if from < end && end <= to {
        Some((start, from))
    } else {
        Some((start, end))
    }
}

/// where a pixel's bytes are in a buffer of guest memory, from its start.
fn pixel_offset(tiled: bool, x: u32, y: u32, width: u32, bpp: u32) -> usize {
    if tiled {
        morton_offset(x, y, width, bpp) as usize
    } else {
        ((y * width + x) * bpp) as usize
    }
}

/// a screen's newest picture turned upright into an image that a presenter
/// sharing the device draws straight from, and the batch that drew it. one
/// is enough, every batch starts with a barrier that waits for the presents
/// submitted before it, so no picture is drawn over while one shows it.
struct Upright {
    image: Image,
    batch: u64,
    /// nothing changed the surface since.
    current: bool,
}

/// a copy of a surface on its way to the host, recorded along with what
/// changed the surface so reading it back needs no batch of its own.
struct Capture {
    buffer: Buffer,
    /// the batch that fills it.
    batch: u64,
    /// nothing changed the surface since.
    current: bool,
    /// a screen shows the surface, so each picture is read out of the
    /// buffer as its batch finishes, before a later one fills it again.
    watched: bool,
    /// the last pictures read out, and the batches that drew them.
    pictures: VecDeque<(u64, Arc<Vec<u8>>)>,
}

/// kinds of work the GPU does, to tell where its time goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Work {
    Other,
    Draw,
    Upload,
    Clear,
    Transfer,
    Capture,
    Copy,
    Download,
    Barrier,
}

const WORK_NAMES: [&str; 9] = ["other", "draw", "upload", "clear", "transfer", "capture", "copy", "download", "barrier"];

/// what the pipeline statistics count, in the order the results come.
const STATISTICS: vk::QueryPipelineStatisticFlags = vk::QueryPipelineStatisticFlags::from_raw(
    vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS.as_raw()
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES.as_raw()
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS.as_raw()
        | vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS.as_raw(),
);
/// timestamps a batch can take.
const TIMESTAMPS: u32 = 4096;
/// batches the times are added up over before they are logged.
const TIMED_BATCHES: u32 = 600;

/// the GPU's time per kind of work, from timestamps written whenever the
/// work changes kind, ZAKURO_GPU_TIMES=1 turns it on.
struct Timing {
    /// nanoseconds a tick of the GPU's clock lasts.
    period: f64,
    /// each command buffer's timestamps, and the work between them.
    pools: HashMap<vk::CommandBuffer, (vk::QueryPool, Vec<Work>)>,
    current: Work,
    /// milliseconds per kind of work.
    totals: [f64; WORK_NAMES.len()],
    batches: u32,
    barriers: u64,
    renderings: u64,
    draws: u64,
    /// each command buffer's pipeline statistics, when the device keeps
    /// them, and the vertices, primitives past clipping, fragments and
    /// compute invocations they added up to.
    statistics: Option<HashMap<vk::CommandBuffer, vk::QueryPool>>,
    counts: [u64; 4],
    /// milliseconds the host waited for the GPU and how many times, for a
    /// batch to record the next in, then for what it needed done.
    waited: [(f64, u32); 2],
}

/// what the host waited for the GPU for.
#[derive(Clone, Copy)]
enum Wait {
    /// a batch to record in, all of them in flight.
    Room,
    /// work done, to read what it wrote or reuse what it read.
    Done,
}

/// a batch's command buffer, the fence the GPU signals once done with it and
/// the ring it stages in.
struct Frame {
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    ring: Buffer,
    /// the batch it holds while the GPU may still be running it.
    pending: Option<u64>,
}

/// a display transfer's registers, taken apart.
pub(crate) struct Transfer {
    pub(crate) input: u32,
    pub(crate) output: u32,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    /// the pixels written, width and height.
    pub(crate) copy: (u32, u32),
    /// the input pixels averaged into each output pixel, across and down.
    pub(crate) scale: (u32, u32),
    pub(crate) flip: bool,
    pub(crate) input_linear: bool,
    pub(crate) output_tiled: bool,
    pub(crate) input_format: ColorFormat,
    pub(crate) output_format: ColorFormat,
}

/// a compute pipeline, the shader and what it takes.
struct Compute {
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// a buffer only the GPU uses.
struct Local {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

/// a texture copied from a surface the GPU drew.
struct Copied {
    image: Image,
    surface: usize,
    /// the surface's generation when it was copied.
    generation: u64,
    used: u64,
}

struct Texture {
    image: Image,
    /// holds the decoded texels the key points at, so the key stays theirs.
    _texels: Arc<[[u8; 4]]>,
    used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    /// the blend function register, when blending.
    blend: Option<u32>,
    /// the logic op, when one other than copy is in use.
    logic_op: Option<LogicOp>,
    /// red, green, blue and alpha writes, one bit each.
    mask: u32,
    depth: bool,
    vertex: VertexStage,
    /// the fragment shader's specialization constants, what decides its
    /// shape, the combiners, the units, lighting, the alpha test, depth.
    fragment: [u32; FRAGMENT_CONSTANTS],
}

/// the fragment shader's specialization constants, as raster.frag numbers
/// them, the last one making it the generic shader.
const FRAGMENT_CONSTANTS: usize = 30;

/// the generic fragment shader's specialization, which reads what the
/// others have as constants from the draw's registers.
const GENERIC: [u32; FRAGMENT_CONSTANTS] = {
    let mut constants = [0; FRAGMENT_CONSTANTS];
    constants[FRAGMENT_CONSTANTS - 1] = 1;
    constants
};

/// the generic fragment shader for a draw's constants. it keeps whether
/// lighting and the procedural texture are on as constants, a lot of code
/// it goes without when they are off, which costs a lot on a small GPU, and
/// which build of the shader it is.
fn generic(fragment: &[u32; FRAGMENT_CONSTANTS]) -> [u32; FRAGMENT_CONSTANTS] {
    let mut generic = GENERIC;
    generic[25] = fragment[25] & 0x400;
    generic[26] = fragment[26];
    generic[28] = fragment[28] & WRITES_DEPTH;
    generic
}

/// the bit of the depth mode constant for the build of raster.frag that
/// works depth out itself.
const WRITES_DEPTH: u32 = 4;

/// the state draws set as they go rather than their pipelines hold, when
/// the device can, so that draws differing only in it share a pipeline.
/// the generic pipelines are then few enough to make them all at the start,
/// rather than each on the draw that first needs it, which stalled the game
/// for as long as the driver took, up to a third of a second.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Dynamic {
    /// blending on or off, its equation, and the color write mask.
    pub(crate) blend: bool,
    /// the logic op on or off, and which.
    pub(crate) logic_op: bool,
}

/// every generic pipeline a draw can need when the blend state is dynamic,
/// with the vertex stages that draws have.
fn generic_keys(shades: bool) -> Vec<PipelineKey> {
    let vertices: &[VertexStage] = if shades { &[VertexStage::Interpreted, VertexStage::Placed] } else { &[VertexStage::Placed] };
    let mut keys = Vec::new();
    for &vertex in vertices {
        for depth in [true, false] {
            for lighting in [0, 1] {
                for procedural in [0, 0x400] {
                    for writes in [0, WRITES_DEPTH] {
                        let mut fragment = GENERIC;
                        fragment[25] = procedural;
                        fragment[26] = lighting;
                        fragment[28] = writes;
                        keys.push(PipelineKey { blend: None, logic_op: None, mask: 0xF, depth, vertex, fragment });
                    }
                }
            }
        }
    }
    keys
}

/// what the PICA's blend function register asks for.
fn blend_equation(config: u32) -> vk::ColorBlendEquationEXT {
    let factor = |raw: u32| {
        if raw & 0xF == 15 {
            vk::BlendFactor::ONE
        } else {
            // the PICA numbers its factors the way Vulkan does
            vk::BlendFactor::from_raw((raw & 0xF) as i32)
        }
    };
    let equation = |raw: u32| {
        if raw & 7 > 4 {
            vk::BlendOp::ADD
        } else {
            vk::BlendOp::from_raw((raw & 7) as i32)
        }
    };
    vk::ColorBlendEquationEXT::default()
        .color_blend_op(equation(config))
        .alpha_blend_op(equation(config >> 8))
        .src_color_blend_factor(factor(config >> 16))
        .dst_color_blend_factor(factor(config >> 20))
        .src_alpha_blend_factor(factor(config >> 24))
        .dst_alpha_blend_factor(factor(config >> 28))
}

/// the build of raster.frag a pipeline runs, of the one leaving depth to
/// the rasterizer and the one working it out.
fn fragment_module([plain, writing]: [vk::ShaderModule; 2], key: &PipelineKey) -> vk::ShaderModule {
    if key.fragment[28] & WRITES_DEPTH != 0 {
        writing
    } else {
        plain
    }
}

/// what a draw's vertices go through on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VertexStage {
    /// triangles the CPU shaded and placed, raster.vert passing them on.
    Placed,
    /// the title's program interpreted, shade.vert.
    Interpreted,
    /// the title's program translated into a module of its own.
    Translated(vk::ShaderModule),
}

/// a program translated, by its fingerprint, its entry point and where
/// its outputs go, what the translation is made from.
type ProgramKey = (u64, u32, [u32; SEMANTICS]);

/// how far a program's translation got.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Translation {
    /// the compiler is on it.
    Pending,
    /// it stays interpreted.
    Failed,
    Done(vk::ShaderModule),
}

/// work for the compiler thread.
enum Job {
    /// a program to translate.
    Translate(ProgramKey, Arc<Program>),
    /// a pipeline to compile, with its vertex shader's module.
    Compile(PipelineKey, vk::ShaderModule),
}

/// what the compiler thread made of a job.
enum Made {
    /// the SPIR-V and the source it came from, which programs that run
    /// the same share a module by.
    Translated(ProgramKey, Result<(Vec<u32>, String), String>),
    Compiled(PipelineKey, Result<vk::Pipeline, String>),
}

/// the jobs waiting for the compiler threads, translations first, as they
/// are quick and pipelines wait for them, then the generic pipelines made at
/// the start, which draws fall back on, then the pipelines asked for last,
/// as those are what is being drawn now, and whether the threads should stop.
struct Queue {
    translations: VecDeque<Job>,
    generic: VecDeque<Job>,
    compiles: Vec<Job>,
    stopped: bool,
    /// pipelines compiled since the cache was last kept on disk, and when
    /// that was, which a thread with nothing else to do sees to.
    compiled: usize,
    saved: std::time::Instant,
}

/// threads translating programs and compiling pipelines, which takes the
/// driver up to seconds for one. draws go on meanwhile through pipelines
/// that draw the same, rather than stall.
struct Compiler {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    made: mpsc::Receiver<Made>,
    /// the jobs queued and not made yet.
    outstanding: usize,
    /// the pipelines asked for, each only once.
    asked: QuickSet<PipelineKey>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl Compiler {
    /// the threads, compiling with a cache, a layout and a fragment shader
    /// that outlive them. a slow driver takes a long while over a scene's
    /// pipelines, which are drawn through the slower generic shader until
    /// then, so there is one for each core the emulation and the driver
    /// leave, up to three.
    fn start(
        device: ash::Device,
        cache: vk::PipelineCache,
        layout: vk::PipelineLayout,
        fragments: [vk::ShaderModule; 2],
        dynamic: Dynamic,
    ) -> Compiler {
        let waiting = Queue {
            translations: VecDeque::new(),
            generic: VecDeque::new(),
            compiles: Vec::new(),
            stopped: false,
            compiled: 0,
            saved: std::time::Instant::now(),
        };
        let queue = Arc::new((Mutex::new(waiting), Condvar::new()));
        let (sent, made) = mpsc::channel();
        let count = thread::available_parallelism().map_or(1, |cores| cores.get().saturating_sub(2).clamp(1, 3));
        let threads = (0..count)
            .map_while(|_| {
                let (jobs, sent, device) = (queue.clone(), sent.clone(), device.clone());
                thread::Builder::new()
                    .name("shader compiler".into())
                    .spawn(move || compile(&jobs, &sent, &device, cache, layout, fragments, dynamic))
                    .ok()
            })
            .collect();
        Compiler { queue, made, outstanding: 0, asked: QuickSet::default(), threads }
    }

    /// queues a generic pipeline ahead of the pipelines asked for, unless
    /// there are no threads, when it is not made until a draw needs it.
    fn send_generic(&mut self, job: Job) {
        if self.threads.is_empty() {
            return;
        }
        let (queue, ready) = &*self.queue;
        queue.lock().unwrap_or_else(PoisonError::into_inner).generic.push_back(job);
        ready.notify_one();
        self.outstanding += 1;
    }

    /// queues a job for the threads, or gives it back when there are none.
    fn send(&mut self, job: Job) -> Option<Job> {
        if self.threads.is_empty() {
            return Some(job);
        }
        let (queue, ready) = &*self.queue;
        let mut waiting = queue.lock().unwrap_or_else(PoisonError::into_inner);
        match job {
            Job::Translate(..) => waiting.translations.push_back(job),
            Job::Compile(..) => waiting.compiles.push(job),
        }
        ready.notify_one();
        self.outstanding += 1;
        None
    }

    /// waits for the jobs the threads are on and drops the rest, giving
    /// back the pipelines they made that no one took.
    fn finish(&mut self) -> Vec<vk::Pipeline> {
        let (queue, ready) = &*self.queue;
        let mut waiting = queue.lock().unwrap_or_else(PoisonError::into_inner);
        waiting.stopped = true;
        waiting.translations.clear();
        waiting.generic.clear();
        waiting.compiles.clear();
        ready.notify_all();
        drop(waiting);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        let made = self.made.try_iter().filter_map(|made| match made {
            Made::Compiled(_, Ok(pipeline)) => Some(pipeline),
            _ => None,
        });
        made.collect()
    }
}

/// what a compiler thread does until it is told to stop, jobs as they come,
/// and keeping the cache on disk now and then when there are none.
fn compile(
    jobs: &(Mutex<Queue>, Condvar),
    sent: &mpsc::Sender<Made>,
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    fragments: [vk::ShaderModule; 2],
    dynamic: Dynamic,
) {
    let (queue, ready) = jobs;
    let lock = || queue.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        let mut waiting = lock();
        let job = loop {
            if waiting.stopped {
                return;
            }
            if let Some(job) =
                waiting.translations.pop_front().or_else(|| waiting.generic.pop_front()).or_else(|| waiting.compiles.pop())
            {
                break job;
            }
            if waiting.compiled == 0 {
                waiting = ready.wait(waiting).unwrap_or_else(PoisonError::into_inner);
            } else if let Some(left) = SAVE_EVERY.checked_sub(waiting.saved.elapsed()).filter(|left| !left.is_zero()) {
                waiting = ready.wait_timeout(waiting, left).unwrap_or_else(PoisonError::into_inner).0;
            } else {
                // the others find nothing to keep meanwhile
                (waiting.compiled, waiting.saved) = (0, std::time::Instant::now());
                drop(waiting);
                save_pipeline_cache(device, cache);
                waiting = lock();
            }
        };
        drop(waiting);
        let compiles = matches!(job, Job::Compile(..));
        let made = work(job, device, cache, layout, fragments, dynamic);
        lock().compiled += compiles as usize;
        if sent.send(made).is_err() {
            return;
        }
    }
}

/// a job done, on the compiler thread or, when waiting for it, on the
/// caller's.
fn work(
    job: Job,
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    fragments: [vk::ShaderModule; 2],
    dynamic: Dynamic,
) -> Made {
    let started = std::time::Instant::now();
    match job {
        Job::Translate(key, program) => {
            // a panic is a bug in the translation, which the interpreter
            // can cover for
            let source = std::panic::catch_unwind(|| glsl::translate(&program, key.1, &key.2))
                .unwrap_or_else(|_| Err("the translation panicked".to_owned()));
            let made = source.and_then(|source| Ok((glsl::compile(&source)?, source)));
            match &made {
                Ok(_) => log::debug!(
                    target: "zakuro_gpu::programs",
                    "translated program {:016X} from {} in {:.1} ms",
                    key.0,
                    key.1,
                    started.elapsed().as_secs_f64() * 1000.0
                ),
                Err(error) => log::debug!(
                    target: "zakuro_gpu::programs",
                    "program {:016X} from {} stays interpreted, {error}",
                    key.0,
                    key.1
                ),
            }
            Made::Translated(key, made)
        }
        Job::Compile(key, module) => {
            let modules = [module, fragment_module(fragments, &key)];
            let made = compile_pipeline(device, cache, layout, modules, &key, vk::PipelineCreateFlags::empty(), dynamic);
            log::debug!(
                target: "zakuro_gpu::pipelines",
                "compiled a pipeline while drawing went on, in {:.1} ms",
                started.elapsed().as_secs_f64() * 1000.0
            );
            Made::Compiled(key, made)
        }
    }
}

/// a draw pipeline for a key, with its vertex and fragment shader modules.
fn compile_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    [vertex, fragment]: [vk::ShaderModule; 2],
    key: &PipelineKey,
    flags: vk::PipelineCreateFlags,
    dynamic: Dynamic,
) -> Result<vk::Pipeline, String> {
    let entries: Vec<vk::SpecializationMapEntry> = (0..FRAGMENT_CONSTANTS as u32)
        .map(|id| vk::SpecializationMapEntry::default().constant_id(id).offset(id * 4).size(4))
        .collect();
    let constants: Vec<u8> = key.fragment.iter().flat_map(|value| value.to_le_bytes()).collect();
    let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(&constants);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main")
            .specialization_info(&specialization),
    ];
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(VERTEX_SIZE as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attributes: Vec<_> = (0..6)
        .map(|location| {
            vk::VertexInputAttributeDescription::default()
                .location(location)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(location * 16)
        })
        .collect();
    // the shaded pipelines read their vertices out of a storage buffer
    let vertex_input = match key.vertex {
        VertexStage::Placed => vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes),
        _ => vk::PipelineVertexInputStateCreateInfo::default(),
    };
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    // depth comes from the fragment shader, what the CPU sends it has
    // been clipped already, what the GPU shades has z clipped to 0..w.
    // the images run bottom up, so a triangle wound counter-clockwise
    // with y up is wound clockwise to Vulkan, which culls per draw
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(key.vertex == VertexStage::Placed)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    let mut attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::from_raw(key.mask));
    if let Some(config) = key.blend {
        let equation = blend_equation(config);
        attachment = attachment
            .blend_enable(true)
            .color_blend_op(equation.color_blend_op)
            .alpha_blend_op(equation.alpha_blend_op)
            .src_color_blend_factor(equation.src_color_blend_factor)
            .dst_color_blend_factor(equation.dst_color_blend_factor)
            .src_alpha_blend_factor(equation.src_alpha_blend_factor)
            .dst_alpha_blend_factor(equation.dst_alpha_blend_factor);
    }
    let attachments = [attachment];
    let mut blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    if let Some(op) = key.logic_op {
        blend = blend.logic_op_enable(true).logic_op(logic_op(op));
    }
    let mut dynamic_states = vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::DEPTH_COMPARE_OP,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_OP,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::CULL_MODE,
    ];
    if dynamic.blend {
        dynamic_states.extend([
            vk::DynamicState::COLOR_BLEND_ENABLE_EXT,
            vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
            vk::DynamicState::COLOR_WRITE_MASK_EXT,
        ]);
    }
    if dynamic.logic_op {
        dynamic_states.extend([vk::DynamicState::LOGIC_OP_ENABLE_EXT, vk::DynamicState::LOGIC_OP_EXT]);
    }
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [COLOR_FORMAT];
    let depth_format = if key.depth { DEPTH_FORMAT } else { vk::Format::UNDEFINED };
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format)
        .stencil_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .flags(flags)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: every state the create info points at lives until it
    // returns
    unsafe { device.create_graphics_pipelines(cache, &[info], None) }
        .map(|pipelines| pipelines[0])
        .map_err(|(_, error)| format!("could not create a pipeline, {error}"))
}

pub struct Hardware {
    /// the device, maybe the presenter's too, kept until this lets go.
    _shared: Arc<SharedDevice>,
    /// whether the presenter shares the device, and shows screens straight
    /// from the images the GPU turns them upright into.
    direct: bool,
    device: ash::Device,
    push: ash::khr::push_descriptor::Device,
    queue: vk::Queue,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    uniform_alignment: u64,
    storage_alignment: u64,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    /// the batches handed to the GPU that it may still be running, oldest
    /// first.
    in_flight: VecDeque<Frame>,
    /// command buffers, fences and rings free for the next batch.
    free: Vec<Frame>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    depth_fragment_shader: vk::ShaderModule,
    shade_shader: vk::ShaderModule,
    /// whether draws go to shade_shader.
    shades: bool,
    /// whether programs are translated rather than interpreted.
    translates: bool,
    /// what a surface's memory was read into last, the next read's buffer.
    scratch: Vec<u8>,
    /// the programs seen and how far their translation got.
    translated: QuickMap<ProgramKey, Translation>,
    /// the last draw's program and how far its translation got, which the
    /// next draw mostly shares, until the compiler brings anything in.
    last_translation: Option<(ProgramKey, Translation)>,
    /// the modules made, by their source, which programs differing only in
    /// words they never run share, and so their pipelines.
    modules: HashMap<String, vk::ShaderModule>,
    /// whether draws wait for their translations and pipelines rather than
    /// draw the same through others meanwhile.
    waits: bool,
    /// whether draws always go through the generic fragment shader, to
    /// compare the two.
    generic: bool,
    compiler: Compiler,
    /// whether the device does logic ops.
    logic_ops: bool,
    /// the state draws set as they go, and what sets it.
    dynamic: Dynamic,
    blend_state: Option<ash::ext::extended_dynamic_state3::Device>,
    logic_op_state: Option<ash::ext::extended_dynamic_state2::Device>,
    /// where the batch copied programs, by their fingerprints.
    programs: QuickMap<u64, u64>,
    /// the blocks the last draw staged, its uniforms and what else its
    /// vertex shader read.
    uniforms: Block,
    shading: Block,
    pipelines: QuickMap<PipelineKey, vk::Pipeline>,
    /// the last pipeline a draw found made for its own key, which the next
    /// draw mostly shares. a pipeline stays in the map for good.
    last_pipeline: Option<(PipelineKey, vk::Pipeline)>,
    /// made the first time a display transfer runs here.
    transfer: Option<Compute>,
    /// made the first time a texture is read out of a depth buffer, with
    /// where the samples go on the way.
    depth: Option<Compute>,
    /// made the first time a scaled screen is captured.
    upright: Option<Compute>,
    /// the same writing an image, made the first time it is needed.
    upright_image: Option<Compute>,
    samples: Option<Local>,
    samplers: QuickMap<(bool, Wrap, Wrap), vk::Sampler>,
    ring: Buffer,
    used: u64,
    readback: Option<Buffer>,
    surfaces: Vec<Surface>,
    /// by the address of the decoded texels.
    textures: HashMap<usize, Texture>,
    /// textures copied from surfaces, by what they are.
    copies: HashMap<DrawnTexture, Copied>,
    /// what unused texture units sample.
    blank: Image,
    /// the tables' generation and where the batch copied them.
    tables: Option<((u64, u64, u64), u64)>,
    recording: bool,
    /// uploads waiting for a barrier before anything reads them.
    uploads: bool,
    /// the color and depth surfaces being rendered into.
    rendering: Option<(usize, Option<usize>)>,
    batch: u64,
    name: String,
    /// how many times the console's resolution surfaces are drawn at.
    scale: u32,
    /// whether the GPU can blit both surface formats, which scaling needs.
    blits: bool,
    timing: Option<Timing>,
    barriers: u64,
    /// whether anything that writes memory was recorded since the last
    /// barrier, without which another one adds nothing.
    unfenced: bool,
    renderings: u64,
    draws: u64,
    /// the pipelines compiled before, kept on disk between runs, so a
    /// combination of fragment stages seen once does not stall again.
    pipeline_cache: vk::PipelineCache,
}

impl Hardware {
    pub fn new() -> Result<Hardware, String> {
        Hardware::with_device(Arc::new(own_device()?), false)
    }

    /// draws with a device made by render_device, which the presenter may
    /// share, and then shows screens straight from the GPU when direct.
    pub fn with_device(shared: Arc<SharedDevice>, direct: bool) -> Result<Hardware, String> {
        let (instance, device, physical, family) = (shared.instance.clone(), shared.device.clone(), shared.physical, shared.family);
        let (logic_ops, statistics, dynamic) = (shared.logic_ops, shared.statistics, shared.dynamic);
        // SAFETY: the physical device came from this instance
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let push = ash::khr::push_descriptor::Device::new(&instance, &device);

        // SAFETY: everything below is made from the device just created,
        // with create infos that live as long as each call
        unsafe {
            let queue = shared.queue;
            let memory_types = instance.get_physical_device_memory_properties(physical);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(family),
                    None,
                )
                .map_err(vk_error("create a command pool"))?;
            let buffers = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(IN_FLIGHT as u32 + 1),
                )
                .map_err(vk_error("allocate a command buffer"))?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vk_error("create a fence"))?;
            let mut spares = Vec::with_capacity(IN_FLIGHT);
            for &commands in &buffers[1..] {
                let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vk_error("create a fence"))?;
                spares.push((commands, fence));
            }

            // the fragment stages', then the program, what else it reads and
            // the inputs of the vertex shader
            let bindings: Vec<_> = (0..8)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(match binding {
                            0..=2 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            3 => vk::DescriptorType::UNIFORM_BUFFER,
                            _ => vk::DescriptorType::STORAGE_BUFFER,
                        })
                        .descriptor_count(1)
                        .stage_flags(if binding < 5 { vk::ShaderStageFlags::FRAGMENT } else { vk::ShaderStageFlags::VERTEX })
                })
                .collect();
            let set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .map_err(vk_error("create a descriptor set layout"))?;
            let set_layouts = [set_layout];
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
                .map_err(vk_error("create a pipeline layout"))?;
            let module = |spirv: &[u8]| -> Result<vk::ShaderModule, String> {
                let words = ash::util::read_spv(&mut Cursor::new(spirv)).map_err(|e| e.to_string())?;
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                    .map_err(vk_error("create a shader module"))
            };
            let vertex_shader = module(VERTEX_SPIRV)?;
            let fragment_shader = module(FRAGMENT_SPIRV)?;
            let depth_fragment_shader = module(FRAGMENT_DEPTH_SPIRV)?;
            let shade_shader = module(SHADE_SPIRV)?;

            let unmade = || Buffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
                mapped: std::ptr::null_mut(),
                incoherent: false,
            };
            let blits = [COLOR_FORMAT, DEPTH_FORMAT].iter().all(|&format| {
                let features = instance.get_physical_device_format_properties(physical, format).optimal_tiling_features;
                features.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
            });
            let timestamps = instance.get_physical_device_queue_family_properties(physical)[family as usize].timestamp_valid_bits > 0;
            // the driver checks the data is for this GPU and ignores it if not
            let saved = pipeline_cache_path().and_then(|path| std::fs::read(path).ok()).unwrap_or_default();
            let pipeline_cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(&saved), None)
                .or_else(|_| device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None))
                .map_err(vk_error("create a pipeline cache"))?;
            let compiler = Compiler::start(device.clone(), pipeline_cache, layout, [fragment_shader, depth_fragment_shader], dynamic);
            let blend_state = dynamic.blend.then(|| ash::ext::extended_dynamic_state3::Device::new(&instance, &device));
            let logic_op_state = dynamic.logic_op.then(|| ash::ext::extended_dynamic_state2::Device::new(&instance, &device));
            let mut hardware = Hardware {
                ring: unmade(),
                in_flight: VecDeque::with_capacity(IN_FLIGHT),
                free: spares.into_iter().map(|(commands, fence)| Frame { commands, fence, ring: unmade(), pending: None }).collect(),
                blank: Image { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() },
                _shared: shared,
                direct,
                device,
                push,
                queue,
                memory_types,
                uniform_alignment: properties.limits.min_uniform_buffer_offset_alignment.max(16),
                storage_alignment: properties.limits.min_storage_buffer_offset_alignment.max(16),
                pool,
                commands: buffers[0],
                fence,
                set_layout,
                layout,
                vertex_shader,
                fragment_shader,
                depth_fragment_shader,
                shade_shader,
                // shading on the CPU instead, to tell the two apart. an
                // integrated GPU does worse at it than the CPU, its draws
                // read each vertex's inputs and uniforms out of storage
                // buffers, a load after another. ZAKURO_GPU_SHADERS=1 has
                // it shade all the same
                shades: match (std::env::var_os("ZAKURO_CPU_SHADERS"), std::env::var_os("ZAKURO_GPU_SHADERS")) {
                    (Some(_), _) => false,
                    (None, Some(_)) => true,
                    (None, None) => properties.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU,
                },
                // interpreting them all instead, to tell the two apart
                translates: std::env::var_os("ZAKURO_INTERPRET_SHADERS").is_none(),
                scratch: Vec::new(),
                translated: QuickMap::default(),
                last_translation: None,
                modules: HashMap::new(),
                // to draw through the pipeline made for each draw from the
                // first on, as the tests do
                waits: cfg!(test) || std::env::var_os("ZAKURO_WAIT_FOR_SHADERS").is_some(),
                generic: std::env::var_os("ZAKURO_GENERIC_SHADERS").is_some(),
                compiler,
                logic_ops,
                dynamic,
                blend_state,
                logic_op_state,
                programs: QuickMap::default(),
                uniforms: Block::default(),
                shading: Block::default(),
                pipelines: QuickMap::default(),
                last_pipeline: None,
                transfer: None,
                depth: None,
                upright: None,
                upright_image: None,
                samples: None,
                samplers: QuickMap::default(),
                used: 0,
                readback: None,
                surfaces: Vec::new(),
                textures: HashMap::new(),
                copies: HashMap::new(),
                tables: None,
                recording: false,
                uploads: false,
                rendering: None,
                batch: 0,
                name,
                scale: 1,
                blits,
                // the queue has to count time for timestamps to mean anything
                timing: (std::env::var_os("ZAKURO_GPU_TIMES").is_some() && timestamps).then(|| Timing {
                        period: properties.limits.timestamp_period as f64,
                        pools: HashMap::new(),
                        current: Work::Other,
                        totals: [0.0; WORK_NAMES.len()],
                        batches: 0,
                        barriers: 0,
                        renderings: 0,
                        draws: 0,
                        statistics: statistics.then(HashMap::new),
                        counts: [0; 4],
                        waited: [(0.0, 0); 2],
                    }),
                barriers: 0,
                unfenced: true,
                renderings: 0,
                draws: 0,
                pipeline_cache,
            };
            let ring_usage = vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC;
            hardware.ring = hardware.buffer(RING_SIZE, ring_usage, false)?;
            for i in 0..hardware.free.len() {
                hardware.free[i].ring = hardware.buffer(RING_SIZE, ring_usage, false)?;
            }
            hardware.begin()?;
            hardware.blank = hardware.image(
                1,
                1,
                COLOR_FORMAT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
            )?;
            // the generic pipelines, while the title starts. a draw that
            // needs one before it is made makes it itself, as before
            if dynamic.blend && !hardware.waits {
                for key in generic_keys(hardware.shades) {
                    let module = hardware.vertex_module(key.vertex);
                    hardware.compiler.send_generic(Job::Compile(key, module));
                }
            }
            Ok(hardware)
        }
    }

    /// the GPU it draws with.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// whether it runs vertex shaders.
    pub(crate) fn shades(&self) -> bool {
        self.shades
    }

    /// how many times the console's resolution it draws at.
    pub fn scale(&self) -> u32 {
        self.scale
    }

    /// draws at a multiple of the console's resolution, before anything is
    /// drawn, when the GPU can scale surfaces, and says the one it took.
    pub fn set_scale(&mut self, scale: u32) -> u32 {
        if self.surfaces.is_empty() && self.blits {
            self.scale = scale.clamp(1, 4);
        }
        self.scale
    }

    /// a host visible buffer, mapped for as long as it lives. one the host
    /// reads back from is cached, reading uncached memory is slow.
    fn buffer(&self, size: u64, usage: vk::BufferUsageFlags, readback: bool) -> Result<Buffer, String> {
        // SAFETY: plain object creation on our device
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(vk_error("create a buffer"))?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
            let (memory, flags) = if readback {
                self.allocate(requirements, visible | vk::MemoryPropertyFlags::HOST_CACHED)
                    .or_else(|_| self.allocate(requirements, visible | vk::MemoryPropertyFlags::HOST_COHERENT))?
            } else {
                self.allocate(requirements, visible | vk::MemoryPropertyFlags::HOST_COHERENT)?
            };
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(vk_error("bind buffer memory"))?;
            let mapped = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_error("map memory"))? as *mut u8;
            let incoherent = !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            Ok(Buffer { buffer, memory, size, mapped, incoherent })
        }
    }

    /// memory with at least flags, and the flags it really has.
    fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags), String> {
        let types = &self.memory_types.memory_types[..self.memory_types.memory_type_count as usize];
        let index = types
            .iter()
            .enumerate()
            .position(|(i, t)| requirements.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(flags))
            .ok_or("no memory type fits")?;
        // SAFETY: an allocation of a type the device reported
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index as u32),
                None,
            )
        }
        .map_err(vk_error("allocate memory"))?;
        Ok((memory, types[index].property_flags))
    }

    /// an image in device memory, put in the general layout by the batch
    /// being recorded, which it never leaves.
    fn image(&self, width: u32, height: u32, format: vk::Format, usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags) -> Result<Image, String> {
        // SAFETY: plain object creation on our device, and a barrier into
        // the command buffer being recorded
        unsafe {
            let image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(vk_error("create an image"))?;
            let requirements = self.device.get_image_memory_requirements(image);
            let (memory, _) = self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            self.device.bind_image_memory(image, memory, 0).map_err(vk_error("bind image memory"))?;
            let range = vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(1);
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(range),
                    None,
                )
                .map_err(vk_error("create an image view"))?;
            let barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(range)];
            self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().image_memory_barriers(&barrier));
            Ok(Image { image, memory, view })
        }
    }

    fn destroy_image(&self, image: &Image) {
        // SAFETY: only called once the GPU is done with the image
        unsafe {
            self.device.destroy_image_view(image.view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.memory, None);
        }
    }

    fn begin(&mut self) -> Result<(), String> {
        if !self.recording {
            // SAFETY: the command buffer is not in use, the last batch was
            // waited for
            unsafe {
                self.device
                    .begin_command_buffer(
                        self.commands,
                        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(vk_error("begin a command buffer"))?;
            }
            self.recording = true;
            self.start_timing()?;
            // the batch before may still be running
            self.unfenced = true;
            self.barrier();
        }
        Ok(())
    }

    /// stamps the start of a batch, when its time is being measured.
    fn start_timing(&mut self) -> Result<(), String> {
        let commands = self.commands;
        let Some(timing) = &mut self.timing else { return Ok(()) };
        let pool = match timing.pools.get_mut(&commands) {
            Some((pool, marks)) => {
                marks.clear();
                *pool
            }
            None => {
                let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(TIMESTAMPS);
                // SAFETY: a pool of ours, made once per command buffer
                let pool = unsafe { self.device.create_query_pool(&info, None) }.map_err(vk_error("create a query pool"))?;
                timing.pools.insert(commands, (pool, Vec::new()));
                pool
            }
        };
        timing.current = Work::Other;
        // SAFETY: recording, outside rendering
        unsafe {
            self.device.cmd_reset_query_pool(commands, pool, 0, TIMESTAMPS);
            self.device.cmd_write_timestamp2(commands, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
        }
        if let Some(statistics) = &mut timing.statistics {
            let pool = match statistics.get(&commands) {
                Some(&pool) => pool,
                None => {
                    let info = vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PIPELINE_STATISTICS)
                        .query_count(1)
                        .pipeline_statistics(STATISTICS);
                    // SAFETY: a pool of ours, made once per command buffer
                    let pool = unsafe { self.device.create_query_pool(&info, None) }.map_err(vk_error("create a query pool"))?;
                    statistics.insert(commands, pool);
                    pool
                }
            };
            // SAFETY: recording, outside rendering, ended before submitting
            unsafe {
                self.device.cmd_reset_query_pool(commands, pool, 0, 1);
                self.device.cmd_begin_query(commands, pool, 0, vk::QueryControlFlags::empty());
            }
        }
        Ok(())
    }

    /// the GPU starts on another kind of work, the time since the last
    /// stamp goes to the kind before. ended stamps the end of the batch.
    fn mark(&mut self, work: Work, ended: bool) {
        let commands = self.commands;
        let Some(timing) = &mut self.timing else { return };
        if !self.recording || (timing.current == work && !ended) {
            return;
        }
        if let Some((pool, marks)) = timing.pools.get_mut(&commands) {
            if (marks.len() as u32) + 1 < TIMESTAMPS {
                // SAFETY: recording, the query was reset when the batch began
                unsafe { self.device.cmd_write_timestamp2(commands, vk::PipelineStageFlags2::ALL_COMMANDS, *pool, marks.len() as u32 + 1) };
                marks.push(timing.current);
            }
        }
        timing.current = work;
    }

    /// adds up a finished batch's times, and logs them every so many.
    fn add_times(&mut self, commands: vk::CommandBuffer) -> Result<(), String> {
        let barriers = std::mem::take(&mut self.barriers);
        let renderings = std::mem::take(&mut self.renderings);
        let draws = std::mem::take(&mut self.draws);
        let Some(timing) = &mut self.timing else { return Ok(()) };
        let Some((pool, marks)) = timing.pools.get_mut(&commands) else { return Ok(()) };
        if !marks.is_empty() {
            let mut stamps = vec![0u64; marks.len() + 1];
            // SAFETY: the batch finished, its fence was waited for
            unsafe {
                self.device
                    .get_query_pool_results(*pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                    .map_err(vk_error("read timestamps"))?;
            }
            for (i, &work) in marks.iter().enumerate() {
                timing.totals[work as usize] += stamps[i + 1].wrapping_sub(stamps[i]) as f64 * timing.period / 1e6;
            }
            marks.clear();
        }
        if let Some(&pool) = timing.statistics.as_ref().and_then(|statistics| statistics.get(&commands)) {
            let mut values = [[0u64; 4]];
            // SAFETY: the batch finished, its fence was waited for
            unsafe {
                self.device
                    .get_query_pool_results(pool, 0, &mut values, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                    .map_err(vk_error("read pipeline statistics"))?;
            }
            for (count, value) in timing.counts.iter_mut().zip(values[0]) {
                *count += value;
            }
        }
        timing.barriers += barriers;
        timing.renderings += renderings;
        timing.draws += draws;
        timing.batches += 1;
        if timing.batches == TIMED_BATCHES {
            let total: f64 = timing.totals.iter().sum();
            let parts: Vec<String> = WORK_NAMES
                .iter()
                .zip(timing.totals)
                .filter(|&(_, ms)| ms > 0.0)
                .map(|(name, ms)| format!("{name} {ms:.1}"))
                .collect();
            let shaded = match timing.statistics {
                Some(_) => {
                    let millions = timing.counts.map(|count| count as f64 / 1e6);
                    format!(
                        ", shaded {:.2}M vertices, {:.2}M primitives, {:.2}M fragments, {:.2}M compute",
                        millions[0], millions[1], millions[2], millions[3]
                    )
                }
                None => String::new(),
            };
            let [(room, rooms), (done, dones)] = timing.waited;
            log::info!(
                target: "zakuro_gpu::times",
                "GPU over {TIMED_BATCHES} batches: {total:.1} ms, {}, {} barriers, {} render passes, {} draws{shaded}, \
                 the host waited {room:.1} ms for room {rooms} times and {done:.1} ms for work done {dones} times",
                parts.join(", "),
                timing.barriers,
                timing.renderings,
                timing.draws
            );
            timing.waited = [(0.0, 0); 2];
            timing.totals = [0.0; WORK_NAMES.len()];
            timing.batches = 0;
            timing.barriers = 0;
            timing.renderings = 0;
            timing.draws = 0;
            timing.counts = [0; 4];
        }
        Ok(())
    }

    /// makes everything recorded so far visible to everything after it,
    /// unless nothing was recorded since the last time.
    fn barrier(&mut self) {
        if !std::mem::take(&mut self.unfenced) {
            return;
        }
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        self.barriers += 1;
        // what it takes is timed on its own, then the work before goes on
        let work = self.timing.as_ref().map(|timing| timing.current);
        self.mark(Work::Barrier, false);
        // SAFETY: recording is on whenever this is called
        unsafe { self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&barrier)) };
        if let Some(work) = work {
            self.mark(work, false);
        }
    }

    fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            // SAFETY: rendering was begun in this command buffer
            unsafe { self.device.cmd_end_rendering(self.commands) };
        }
    }

    /// room for size bytes in the ring, where it starts.
    fn stage(&mut self, size: u64, align: u64) -> Result<u64, String> {
        let offset = self.used.next_multiple_of(align);
        if offset + size > self.ring.size {
            return Err(format!("a draw needs {size} more bytes than a batch has"));
        }
        self.used = offset + size;
        Ok(offset)
    }

    /// where the batch holds a block's words, staged here unless they are
    /// the words of the last block of its kind, which draws in a row mostly
    /// share, the GPU reading those again. the ring is memory the CPU
    /// writes past its caches, slower than gathering the words and
    /// comparing them, and it takes a run of whole lines best, so the words
    /// go in at once.
    fn stage_block(&mut self, block: &mut Block, size: u64, align: u64) -> Result<u64, String> {
        debug_assert_eq!(block.words.len() as u64 * 4, size);
        if let Some(at) = block.staged() {
            return Ok(at);
        }
        let offset = self.stage(size, align)?;
        for (out, word) in self.ring(offset, size).as_chunks_mut::<4>().0.iter_mut().zip(&block.words) {
            *out = word.to_le_bytes();
        }
        block.staged_at(offset);
        Ok(offset)
    }

    fn ring(&mut self, offset: u64, size: u64) -> &mut [u8] {
        // SAFETY: the ring is mapped for its whole size and stage kept the
        // range inside it, the GPU is not reading it while this batch is
        // being recorded
        unsafe { std::slice::from_raw_parts_mut(self.ring.mapped.add(offset as usize), size as usize) }
    }

    /// the surface for a guest buffer, matching guest memory.
    fn surface<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        size: (u32, u32),
        kind: Kind,
        tiled: bool,
    ) -> Result<usize, String> {
        self.surface_rows(memory, addr, size, kind, tiled, None)
    }

    /// the surface for a guest buffer, matching guest memory over the rows
    /// of memory given, or all of them.
    /// clears rows of memory of a surface to one pixel, on the GPU.
    fn clear_rows(&mut self, index: usize, rows: (u32, u32), pixel: &[u8]) -> Result<(), String> {
        let surface = &self.surfaces[index];
        let (width, height, kind, view) = (surface.width, surface.height, surface.kind, surface.image.view);
        let scale = self.scale;
        self.begin()?;
        self.mark(Work::Clear, false);
        self.renderings += 1;
        self.end_rendering();
        self.barrier();
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: width * scale, height: height * scale } };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(1);
        let (info, clear) = match kind {
            Kind::Color(format) => {
                let rgba = format.decode(pixel).map(|c| c as f32 / 255.0);
                let clear = vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: rgba } });
                (info.color_attachments(&attachment), clear)
            }
            Kind::Depth(sample) => {
                let (depth, stencil) = match sample {
                    2 => (((u16::from_le_bytes([pixel[0], pixel[1]]) as u64 * 0xFF_FFFF + 0x7FFF) / 0xFFFF) as u32, 0),
                    3 => (u32::from_le_bytes([pixel[0], pixel[1], pixel[2], 0]), 0),
                    _ => (u32::from_le_bytes([pixel[0], pixel[1], pixel[2], 0]), pixel[3] as u32),
                };
                let value = vk::ClearDepthStencilValue { depth: depth as f32 / 16_777_215.0, stencil };
                let clear = vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .clear_value(vk::ClearValue { depth_stencil: value });
                (info.depth_attachment(&attachment[0]).stencil_attachment(&attachment[0]), clear)
            }
        };
        // rows of memory run down from the window's top, which is the
        // image's last row
        let rect = [vk::ClearRect::default()
            .rect(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: ((height - rows.1) * scale) as i32 },
                extent: vk::Extent2D { width: width * scale, height: (rows.1 - rows.0) * scale },
            })
            .layer_count(1)];
        // SAFETY: recording, outside rendering, on an image in the general
        // layout made to be rendered into, the rectangle inside it
        unsafe {
            self.unfenced = true;
            self.device.cmd_begin_rendering(self.commands, &info);
            self.device.cmd_clear_attachments(self.commands, &[clear], &rect);
            self.device.cmd_end_rendering(self.commands);
        }
        self.uploads = true;
        Ok(())
    }

    /// the height of the tallest surface of a kind at an address and width,
    /// at least height, the one a draw of that height goes into.
    fn taller(&self, addr: u32, kind: Kind, width: u32, height: u32) -> u32 {
        self.surfaces
            .iter()
            .filter(|s| s.addr == addr && s.kind == kind && s.tiled && s.width == width)
            .map(|s| s.height)
            .fold(height, u32::max)
    }

    fn surface_rows<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        (width, height): (u32, u32),
        kind: Kind,
        tiled: bool,
        rows: Option<(u32, u32)>,
    ) -> Result<usize, String> {
        let index = match self
            .surfaces
            .iter()
            .position(|s| s.addr == addr && s.width == width && s.height == height && s.kind == kind && s.tiled == tiled)
        {
            Some(index) => index,
            None => {
                let (format, usage, aspect) = match kind {
                    Kind::Color(_) => (
                        COLOR_FORMAT,
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
                        vk::ImageAspectFlags::COLOR,
                    ),
                    Kind::Depth(_) => (
                        DEPTH_FORMAT,
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    ),
                };
                let usage = usage | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
                let image = self.image(width * self.scale, height * self.scale, format, usage, aspect)?;
                let native = match self.scale {
                    1 => None,
                    _ => {
                        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
                        Some(self.image(width, height, format, usage, aspect)?)
                    }
                };
                self.uploads = true;
                self.surfaces.push(Surface {
                    image,
                    addr,
                    width,
                    height,
                    kind,
                    tiled,
                    shadow: Vec::new(),
                    pixel: None,
                    checked: false,
                    dirty: None,
                    guarded: None,
                    write_guarded: None,
                    cleared: None,
                    capture: None,
                    native,
                    screen: None,
                    upright: None,
                    generation: 0,
                });
                self.surfaces.len() - 1
            }
        };
        if self.surfaces[index].checked {
            return Ok(index);
        }
        // what another surface over the same memory drew has to reach guest
        // memory before this one reads it, where this one is used
        let size = self.surfaces[index].size();
        let row = self.surfaces[index].row_bytes();
        let (from, to) = rows.map_or((addr, size), |(start, end)| (addr + start * row, (end - start) * row));
        let others: Vec<usize> = (0..self.surfaces.len())
            .filter(|&i| i != index && self.surfaces[i].dirty_overlaps(from, to))
            .collect();
        if !others.is_empty() {
            self.write_back(memory, others)?;
            self.begin()?;
        }
        // into the last lookup's buffer, a surface's worth of bytes allocated
        // for each costs more than reading them. compared in place where
        // memory is in one piece, most lookups find nothing changed
        let mut bytes = std::mem::take(&mut self.scratch);
        let whole = 0..size as usize;
        let same = memory.slice(addr, size as usize).is_some_and(|now| self.surfaces[index].shadows(whole.clone(), now));
        if !same {
            bytes.resize(size as usize, 0);
            memory.read(addr, &mut bytes);
        }
        if !same && !self.surfaces[index].shadows(whole, &bytes) {
            if self.surfaces[index].dirty.is_some() {
                // memory changed beside rows the GPU drew and it lacks, which
                // an upload alone would lose. they come down first, along
                // with what changed
                self.write_back(memory, vec![index])?;
                self.begin()?;
                memory.read(addr, &mut bytes);
            }
            self.upload(index, &bytes)?;
            self.surfaces[index].keep(&bytes);
        }
        self.scratch = bytes;
        self.surfaces[index].checked = true;
        Ok(index)
    }

    /// the GPU drew rows of a surface, which other surfaces over the same
    /// memory hold older, the next use of one looks at memory again, after
    /// what this one drew is written back. the other buffer of the same draw
    /// is left alone, titles pack color and depth next to each other.
    fn overdrawn(&mut self, index: usize, (start, end): (u32, u32), pair: Option<usize>) {
        let s = &self.surfaces[index];
        let row = s.row_bytes();
        // whole rows of tiles, as a draw changes tiled memory
        let (start, end) = (start / 8 * 8, end.div_ceil(8).saturating_mul(8).min(s.height));
        let (addr, len) = (s.addr + start * row, end.saturating_sub(start) * row);
        for i in 0..self.surfaces.len() {
            if i != index && Some(i) != pair && self.surfaces[i].overlaps(addr, len) {
                self.surfaces[i].checked = false;
            }
        }
    }

    /// copies a guest buffer's bytes into its surface.
    fn upload(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        self.end_rendering();
        self.mark(Work::Upload, false);
        let (width, height, kind, tiled) = {
            let s = &self.surfaces[index];
            (s.width, s.height, s.kind, s.tiled)
        };
        let pixels = (width * height) as u64;
        let image = self.surfaces[index].native.as_ref().unwrap_or(&self.surfaces[index].image).image;
        let extent = vk::Extent3D { width, height, depth: 1 };
        let region = |offset: u64, aspect: vk::ImageAspectFlags| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1))
                .image_extent(extent)
        };
        match kind {
            Kind::Color(format) => {
                let offset = self.stage(pixels * 4, 16)?;
                let bpp = format.bytes_per_pixel();
                let staging = self.ring(offset, pixels * 4);
                for y in 0..height {
                    for x in 0..width {
                        let at = pixel_offset(tiled, x, y, width, bpp as u32);
                        let rgba = format.decode(&bytes[at..at + bpp]);
                        let out = (((height - 1 - y) * width + x) * 4) as usize;
                        staging[out..out + 4].copy_from_slice(&rgba);
                    }
                }
                let regions = [region(offset, vk::ImageAspectFlags::COLOR)];
                // SAFETY: recording, and the regions lie inside the ring
                unsafe {
                    self.unfenced = true;
                    self.device.cmd_copy_buffer_to_image(self.commands, self.ring.buffer, image, vk::ImageLayout::GENERAL, &regions)
                };
            }
            Kind::Depth(sample) => {
                let depths = self.stage(pixels * 4, 16)?;
                let stencils = self.stage(pixels, 16)?;
                let mut values = vec![0u32; pixels as usize];
                let mut stencil = vec![0u8; pixels as usize];
                for y in 0..height {
                    for x in 0..width {
                        let at = morton_offset(x, y, width, sample) as usize;
                        let i = ((height - 1 - y) * width + x) as usize;
                        match sample {
                            2 => {
                                let d16 = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u64;
                                values[i] = ((d16 * 0xFF_FFFF + 0x7FFF) / 0xFFFF) as u32;
                            }
                            3 => values[i] = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], 0]),
                            _ => {
                                let word = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
                                values[i] = word & 0xFF_FFFF;
                                stencil[i] = (word >> 24) as u8;
                            }
                        }
                    }
                }
                let staging = self.ring(depths, pixels * 4);
                for (out, value) in staging.as_chunks_mut::<4>().0.iter_mut().zip(&values) {
                    *out = value.to_le_bytes();
                }
                self.ring(stencils, pixels).copy_from_slice(&stencil);
                let regions = [region(depths, vk::ImageAspectFlags::DEPTH), region(stencils, vk::ImageAspectFlags::STENCIL)];
                // SAFETY: as above
                unsafe {
                    self.unfenced = true;
                    self.device.cmd_copy_buffer_to_image(self.commands, self.ring.buffer, image, vk::ImageLayout::GENERAL, &regions)
                };
            }
        }
        if self.scale > 1 {
            self.barrier();
            self.blit(index, true);
        }
        self.uploads = true;
        self.surfaces[index].replaced();
        Ok(())
    }

    /// records a blit between a scaled surface and its image at the
    /// console's resolution, up or down, taking the nearest pixel either way.
    fn blit(&mut self, index: usize, up: bool) {
        let s = &self.surfaces[index];
        let Some(native) = &s.native else { return };
        let aspect = match s.kind {
            Kind::Color(_) => vk::ImageAspectFlags::COLOR,
            Kind::Depth(_) => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        };
        let layers = vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1);
        let corner = |width: u32, height: u32| [vk::Offset3D::default(), vk::Offset3D { x: width as i32, y: height as i32, z: 1 }];
        let small = corner(s.width, s.height);
        let big = corner(s.width * self.scale, s.height * self.scale);
        let (from, to, from_corner, to_corner) =
            if up { (native.image, s.image.image, small, big) } else { (s.image.image, native.image, big, small) };
        let region = [vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets(from_corner)
            .dst_subresource(layers)
            .dst_offsets(to_corner)];
        // SAFETY: recording, outside rendering, between two images of the
        // surface's format in the general layout made for transfers
        unsafe {
            self.unfenced = true;
            self.device.cmd_blit_image(
                self.commands,
                from,
                vk::ImageLayout::GENERAL,
                to,
                vk::ImageLayout::GENERAL,
                &region,
                vk::Filter::NEAREST,
            )
        };
    }

    /// the texture's image, uploaded the first time it is drawn with.
    fn texture(&mut self, bound: &BoundTexture) -> Result<vk::ImageView, String> {
        let key = Arc::as_ptr(&bound.texels) as *const u8 as usize;
        if let Some(texture) = self.textures.get_mut(&key) {
            texture.used = self.batch;
            return Ok(texture.image.view);
        }
        self.mark(Work::Upload, false);
        self.end_rendering();
        let image = self.image(
            bound.width,
            bound.height,
            COLOR_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
        )?;
        let size = bound.texels.len() as u64 * 4;
        let offset = self.stage(size, 16)?;
        let staging = self.ring(offset, size);
        for (out, texel) in staging.as_chunks_mut::<4>().0.iter_mut().zip(bound.texels.iter()) {
            *out = *texel;
        }
        let regions = [vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: bound.width, height: bound.height, depth: 1 })];
        // SAFETY: recording, the region lies inside the ring and the image
        unsafe {
            self.unfenced = true;
            self.device.cmd_copy_buffer_to_image(self.commands, self.ring.buffer, image.image, vk::ImageLayout::GENERAL, &regions)
        };
        self.uploads = true;
        let view = image.view;
        self.textures.insert(key, Texture { image, _texels: bound.texels.clone(), used: self.batch });
        Ok(view)
    }

    fn sampler(&mut self, linear: bool, s: Wrap, t: Wrap) -> Result<vk::Sampler, String> {
        if let Some(&sampler) = self.samplers.get(&(linear, s, t)) {
            return Ok(sampler);
        }
        // the border is the shader's to draw, past the edge it clamps
        let mode = |wrap: Wrap| match wrap {
            Wrap::Repeat => vk::SamplerAddressMode::REPEAT,
            Wrap::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };
        let filter = if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(mode(s))
            .address_mode_v(mode(t))
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        // SAFETY: plain object creation on our device
        let sampler = unsafe { self.device.create_sampler(&info, None) }.map_err(vk_error("create a sampler"))?;
        self.samplers.insert((linear, s, t), sampler);
        Ok(sampler)
    }

    /// where the batch holds the current lighting tables.
    fn tables(&mut self, tables: &Tables, procedural: &proctex::Tables, fog: &fog::Table) -> Result<u64, String> {
        let generation = (tables.generation(), procedural.generation(), fog.generation());
        if let Some((staged, offset)) = self.tables {
            if staged == generation {
                return Ok(offset);
            }
        }
        let offset = self.stage(TABLES_SIZE, self.storage_alignment)?;
        let staging = self.ring(offset, TABLES_SIZE);
        let (colors, steps) = procedural.colors();
        // a plain loop over each, which goes as fast as copying, a chain of
        // flattened iterators took a quarter of a frame
        let parts = [
            tables.entries().as_flattened().as_flattened(),
            procedural.maps().as_flattened().as_flattened(),
            colors.as_flattened(),
            steps.as_flattened(),
            fog.entries().as_flattened(),
        ];
        let mut out = staging.as_chunks_mut::<4>().0;
        for part in parts {
            let (here, rest) = out.split_at_mut(part.len());
            for (bytes, value) in here.iter_mut().zip(part) {
                *bytes = value.to_le_bytes();
            }
            out = rest;
        }
        self.tables = Some((generation, offset));
        Ok(offset)
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<vk::Pipeline, String> {
        if let Some(&pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline);
        }
        let modules = [self.vertex_module(key.vertex), fragment_module(self.fragment_modules(), &key)];
        let started = std::time::Instant::now();
        let pipeline = compile_pipeline(
            &self.device,
            self.pipeline_cache,
            self.layout,
            modules,
            &key,
            vk::PipelineCreateFlags::empty(),
            self.dynamic,
        )?;
        log::debug!(
            target: "zakuro_gpu::pipelines",
            "compiled pipeline {} on the draw in {:.1} ms",
            self.pipelines.len() + 1,
            started.elapsed().as_secs_f64() * 1000.0
        );
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// the pipeline for a key when the cache on disk has it, made without
    /// compiling anything.
    fn cached(&mut self, key: PipelineKey) -> Option<vk::Pipeline> {
        let modules = [self.vertex_module(key.vertex), fragment_module(self.fragment_modules(), &key)];
        let started = std::time::Instant::now();
        let flags = vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
        let pipeline = compile_pipeline(&self.device, self.pipeline_cache, self.layout, modules, &key, flags, self.dynamic).ok()?;
        log::debug!(
            target: "zakuro_gpu::pipelines",
            "took pipeline {} from the cache in {:.1} ms",
            self.pipelines.len() + 1,
            started.elapsed().as_secs_f64() * 1000.0
        );
        self.pipelines.insert(key, pipeline);
        Some(pipeline)
    }

    /// the pipeline a draw goes through. until the one made for its key is
    /// ready the compiler makes it, and the draw goes through one that
    /// draws the same, the interpreter's with the same fragment stages when
    /// there is one, or else the generic fragment shader's, made right away
    /// once per blend function and the like. a program still being
    /// translated does not have its interpreted pipeline asked for, it
    /// would not be used for long.
    fn pipeline_for(&mut self, key: PipelineKey, translating: bool) -> Result<vk::Pipeline, String> {
        let vertex = match key.vertex {
            VertexStage::Placed => VertexStage::Placed,
            _ => VertexStage::Interpreted,
        };
        let generic = PipelineKey { vertex, fragment: generic(&key.fragment), ..key };
        if self.generic {
            return self.pipeline(generic);
        }
        if let Some((last, pipeline)) = self.last_pipeline {
            if last == key {
                return Ok(pipeline);
            }
        }
        if let Some(&pipeline) = self.pipelines.get(&key) {
            self.last_pipeline = Some((key, pipeline));
            return Ok(pipeline);
        }
        if !translating && self.compiler.asked.insert(key) {
            if let Some(pipeline) = self.cached(key) {
                return Ok(pipeline);
            }
            // made right here when waiting for it
            self.hand(Job::Compile(key, self.vertex_module(key.vertex)));
            if let Some(&pipeline) = self.pipelines.get(&key) {
                return Ok(pipeline);
            }
        }
        // meanwhile, or for good when the driver turned it down
        if let VertexStage::Translated(_) = key.vertex {
            let interpreted = PipelineKey { vertex: VertexStage::Interpreted, ..key };
            if self.waits || self.pipelines.contains_key(&interpreted) {
                return self.pipeline(interpreted);
            }
        }
        self.pipeline(generic)
    }

    /// raster.frag leaving depth to the rasterizer, and working it out.
    fn fragment_modules(&self) -> [vk::ShaderModule; 2] {
        [self.fragment_shader, self.depth_fragment_shader]
    }

    /// the module a vertex stage runs.
    fn vertex_module(&self, vertex: VertexStage) -> vk::ShaderModule {
        match vertex {
            VertexStage::Placed => self.vertex_shader,
            VertexStage::Interpreted => self.shade_shader,
            VertexStage::Translated(module) => module,
        }
    }

    /// a memory fill wrote bytes at addr, a pattern of up to four bytes over
    /// and over. a surface it covered with one value is cleared to it on the
    /// GPU as well, rather than uploaded again the next time it is drawn
    /// into.
    pub(crate) fn filled(&mut self, addr: u32, bytes: &[u8]) -> Result<(), String> {
        let end = addr as u64 + bytes.len() as u64;
        for index in 0..self.surfaces.len() {
            let surface = &self.surfaces[index];
            let size = surface.size() as usize;
            if let Some(rows) = rows_filled(surface, addr, bytes.len() as u32) {
                // whole rows of tiles of it, cleared on the GPU, the rest of
                // what it drew stays there
                let row = surface.row_bytes() as usize;
                let at = (surface.addr as usize + rows.0 as usize * row) - addr as usize;
                let filled = &bytes[at..at + (rows.1 - rows.0) as usize * row];
                let bpp = surface.kind.bytes() as usize;
                if filled.iter().enumerate().take(12).all(|(i, &b)| b == filled[i % bpp]) {
                    let pixel = filled[..bpp].to_vec();
                    self.clear_rows(index, rows, &pixel)?;
                    let surface = &mut self.surfaces[index];
                    surface.keep_rows(rows.0 as usize * row..rows.1 as usize * row, &pixel);
                    // memory holds those rows as the image does now, what
                    // the GPU drew that has not come down is the rest. a
                    // buffer of another shape drawn over the filled rows
                    // does not have this one written back first, Inazuma
                    // Eleven GO draws both screens through one every frame
                    let left = surface.dirty.and_then(|dirty| without(dirty, rows));
                    if left != surface.dirty {
                        surface.dirty = left;
                        // the next draw guards what it draws again
                        surface.guarded = None;
                        surface.write_guarded = None;
                    }
                    if left.is_some() {
                        surface.cleared = Some(surface.cleared.map_or(rows, |(from, to)| (from.min(rows.0), to.max(rows.1))));
                    }
                    surface.replaced();
                } else {
                    self.surfaces[index].checked = false;
                }
                continue;
            }
            if surface.addr < addr || surface.addr as u64 + size as u64 > end {
                continue;
            }
            let offset = (surface.addr - addr) as usize;
            let filled = &bytes[offset..offset + size];
            let bpp = surface.kind.bytes() as usize;
            // a pattern that does not line up with the pixels leaves them
            // different from each other, and uploading handles that, what
            // the GPU drew is gone
            if !filled.iter().enumerate().take(12).all(|(i, &b)| b == filled[i % bpp]) {
                let surface = &mut self.surfaces[index];
                // the image is not what memory holds, even when the fill left
                // memory as the shadow had it
                surface.shadow.clear();
                surface.pixel = None;
                surface.dirty = None;
                surface.guarded = None;
                surface.write_guarded = None;
                surface.cleared = None;
                surface.checked = false;
                continue;
            }
            let pixel = filled[..bpp].to_vec();
            let (image, kind) = (surface.image.image, surface.kind);
            self.begin()?;
            self.mark(Work::Clear, false);
            self.end_rendering();
            // SAFETY: recording, outside rendering, on an image in the
            // general layout that allows transfers into it
            unsafe {
                match kind {
                    Kind::Color(format) => {
                        let rgba = format.decode(&pixel);
                        let color = vk::ClearColorValue { float32: rgba.map(|c| c as f32 / 255.0) };
                        let range = [vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)];
                        self.unfenced = true;
                        self.device.cmd_clear_color_image(self.commands, image, vk::ImageLayout::GENERAL, &color, &range);
                    }
                    Kind::Depth(sample) => {
                        let (depth, stencil) = match sample {
                            2 => {
                                let d16 = u16::from_le_bytes([pixel[0], pixel[1]]) as u64;
                                (((d16 * 0xFF_FFFF + 0x7FFF) / 0xFFFF) as u32, 0)
                            }
                            3 => (u32::from_le_bytes([pixel[0], pixel[1], pixel[2], 0]), 0),
                            _ => (u32::from_le_bytes([pixel[0], pixel[1], pixel[2], 0]), pixel[3] as u32),
                        };
                        let value = vk::ClearDepthStencilValue { depth: depth as f32 / 16_777_215.0, stencil };
                        let range = [vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                            .level_count(1)
                            .layer_count(1)];
                        self.unfenced = true;
                        self.device.cmd_clear_depth_stencil_image(self.commands, image, vk::ImageLayout::GENERAL, &value, &range);
                    }
                }
            }
            self.uploads = true;
            let surface = &mut self.surfaces[index];
            // the pixel stands in for the bytes rather than a copy of them
            surface.keep_pixel(&pixel);
            surface.dirty = None;
            surface.guarded = None;
            surface.write_guarded = None;
            surface.cleared = None;
            surface.checked = false;
            surface.replaced();
        }
        Ok(())
    }

    /// makes sure guest memory holds what the GPU drew over a range, before
    /// something reads it.
    pub(crate) fn prepare_read<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        if self.surfaces.iter().any(|s| s.dirty_overlaps(addr, len)) {
            self.sync(memory, addr, len)?;
        }
        Ok(())
    }

    /// a fill is about to write a range, whatever the GPU drew over the part
    /// of a buffer it leaves alone has to come down first.
    pub(crate) fn before_fill<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32, width: u32) -> Result<(), String> {
        let end = addr as u64 + len as u64;
        // whole rows of tiles get cleared on the GPU, when the pattern lines
        // up with the pixels
        let cleared = |s: &Surface| s.kind.bytes().is_multiple_of(width) && rows_filled(s, addr, len).is_some();
        let partial: Vec<usize> = (0..self.surfaces.len())
            .filter(|&i| {
                let s = &self.surfaces[i];
                s.dirty_overlaps(addr, len) && (s.addr < addr || s.addr as u64 + s.size() as u64 > end) && !cleared(s)
            })
            .collect();
        if !partial.is_empty() {
            self.write_back(memory, partial)?;
        }
        Ok(())
    }

    /// copies the vertices the CPU shaded and placed into the batch, with
    /// the target's pixels as clip space and w kept for perspective, then
    /// the indices of the triangles, and says where both start.
    fn place(&mut self, vertices: &[Screen], indices: &[u32], (width, height): (f32, f32), depth_map: DepthMap) -> Result<(u64, u64), String> {
        let index_bytes = (indices.len() * 4) as u64;
        let index_offset = self.stage(index_bytes, 4)?;
        let staging = self.ring(index_offset, index_bytes);
        for (out, index) in staging.as_chunks_mut::<4>().0.iter_mut().zip(indices) {
            *out = index.to_le_bytes();
        }
        let vertex_bytes = (vertices.len() * VERTEX_SIZE) as u64;
        let vertex_offset = self.stage(vertex_bytes, 16)?;
        // a pixel whose center sits exactly on an edge goes to the triangle
        // on its right, or above it for a flat edge, on the PICA, and in
        // Vulkan to the one on its right or further down the image, so the
        // images run bottom up
        let staging = self.ring(vertex_offset, vertex_bytes);
        for (out, v) in staging.as_chunks_mut::<VERTEX_SIZE>().0.iter_mut().zip(vertices) {
            let w = 1.0 / v.inv_w;
            let x = v.x / width * 2.0 - 1.0;
            let y = v.y / height * 2.0 - 1.0;
            let depth = v.z * depth_map.scale + depth_map.offset;
            let (c, t, q, view) = (v.color_over_w, v.texcoords_over_w, v.quaternion_over_w, v.view_over_w);
            let values: [f32; 24] = [
                x * w,
                y * w,
                0.0,
                w,
                c[0] * w,
                c[1] * w,
                c[2] * w,
                c[3] * w,
                t[0][0] * w,
                t[0][1] * w,
                t[1][0] * w,
                t[1][1] * w,
                t[2][0] * w,
                t[2][1] * w,
                depth,
                0.0,
                q[0] * w,
                q[1] * w,
                q[2] * w,
                q[3] * w,
                view[0] * w,
                view[1] * w,
                view[2] * w,
                0.0,
            ];
            for (bytes, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *bytes = value.to_le_bytes();
            }
        }
        Ok((vertex_offset, index_offset))
    }

    /// whether programs get translated, for comparing the two, and whether
    /// draws wait for that.
    #[cfg(test)]
    pub(super) fn set_translates(&mut self, translates: bool, waits: bool) {
        self.translates = translates;
        self.waits = waits;
    }

    /// whether draws always go through the generic fragment shader.
    #[cfg(test)]
    pub(super) fn set_generic(&mut self, generic: bool) {
        self.generic = generic;
    }

    /// whether the device does logic ops, without which draws leave them out.
    #[cfg(test)]
    pub(super) fn logic_ops(&self) -> bool {
        self.logic_ops
    }

    /// the programs that run translated so far.
    #[cfg(test)]
    pub(super) fn translations(&self) -> usize {
        self.translated.values().filter(|translation| matches!(translation, Translation::Done(_))).count()
    }

    /// the pipelines made for translated programs so far.
    #[cfg(test)]
    pub(super) fn translated_pipelines(&self) -> usize {
        self.pipelines.keys().filter(|key| matches!(key.vertex, VertexStage::Translated(_))).count()
    }

    /// the stage a draw's program runs in, and whether it is still being
    /// translated. the first time it is seen the compiler starts translating
    /// it, and it is interpreted until that is done, and for good if it
    /// cannot be.
    fn vertex_stage(&mut self, shading: &Shading) -> (VertexStage, bool) {
        if !self.translates {
            return (VertexStage::Interpreted, false);
        }
        let unit = shading.unit;
        let key = (unit.fingerprint(), unit.entry_point, shading.semantics);
        let translation = match self.last_translation {
            Some((last, translation)) if last == key => translation,
            _ => self.translation(unit, key),
        };
        self.last_translation = Some((key, translation));
        match translation {
            Translation::Pending => (VertexStage::Interpreted, true),
            Translation::Failed => (VertexStage::Interpreted, false),
            Translation::Done(module) => (VertexStage::Translated(module), false),
        }
    }

    /// how far a program's translation got, the compiler starting on it the
    /// first time the program is seen.
    fn translation(&mut self, unit: &ShaderUnit, key: ProgramKey) -> Translation {
        if let Some(&translation) = self.translated.get(&key) {
            return translation;
        }
        match unit.prepared() {
            Some(program) => {
                self.translated.insert(key, Translation::Pending);
                self.hand(Job::Translate(key, program));
            }
            None => {
                self.translated.insert(key, Translation::Failed);
                log::debug!(
                    target: "zakuro_gpu::programs",
                    "program {:016X} from {} stays interpreted, it was not prepared",
                    key.0,
                    key.1
                );
            }
        }
        self.translated[&key]
    }

    /// has the compiler do a job, or does it here, when waiting for it or
    /// when there is no compiler.
    fn hand(&mut self, job: Job) {
        let job = match self.waits {
            true => job,
            false => match self.compiler.send(job) {
                Some(job) => job,
                None => return,
            },
        };
        let made = work(job, &self.device, self.pipeline_cache, self.layout, self.fragment_modules(), self.dynamic);
        self.take(made);
    }

    /// takes in what the compiler made since last time.
    fn collect(&mut self) {
        while self.compiler.outstanding > 0 {
            let Ok(made) = self.compiler.made.try_recv() else {
                break;
            };
            self.compiler.outstanding -= 1;
            self.take(made);
        }
    }

    fn take(&mut self, made: Made) {
        // a translation may have come in or gone back to the interpreter
        self.last_translation = None;
        match made {
            Made::Translated(key, Ok((words, source))) => {
                let module = match self.modules.get(&source) {
                    Some(&module) => module,
                    None => {
                        let info = vk::ShaderModuleCreateInfo::default().code(&words);
                        // SAFETY: the words are SPIR-V naga made and validated
                        match unsafe { self.device.create_shader_module(&info, None) } {
                            Ok(module) => {
                                self.modules.insert(source, module);
                                module
                            }
                            Err(error) => {
                                log::warn!("could not create a translated program's module, {error}, interpreting it");
                                self.translated.insert(key, Translation::Failed);
                                return;
                            }
                        }
                    }
                };
                self.translated.insert(key, Translation::Done(module));
            }
            // the compiler said why
            Made::Translated(key, Err(_)) => {
                self.translated.insert(key, Translation::Failed);
            }
            Made::Compiled(key, Ok(pipeline)) => match self.pipelines.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(pipeline);
                }
                // a draw made it meanwhile
                // SAFETY: nothing recorded uses the one just made
                Entry::Occupied(_) => unsafe { self.device.destroy_pipeline(pipeline, None) },
            },
            Made::Compiled(key, Err(error)) => match key.vertex {
                // a translated program the driver turns down goes back to
                // the interpreter
                VertexStage::Translated(failed) => {
                    log::warn!("a translated vertex shader could not be used, {error}, interpreting it");
                    for translation in self.translated.values_mut() {
                        if *translation == Translation::Done(failed) {
                            *translation = Translation::Failed;
                        }
                    }
                }
                // the generic fragment shader goes on drawing it
                _ => log::warn!("could not compile a pipeline, {error}, drawing through the generic one"),
            },
        }
    }

    /// copies what the vertex shader reads into the batch, and says where
    /// the program, the rest of what it reads, the inputs and the indices
    /// went.
    fn stage_shading<M: GpuMemory>(&mut self, memory: &mut M, shading: &Shading, depth_map: DepthMap) -> Result<[(u64, u64); 4], String> {
        let unit = shading.unit;
        let fingerprint = unit.fingerprint();
        let program = match self.programs.get(&fingerprint) {
            Some(&offset) => offset,
            None => {
                let offset = self.stage(PROGRAM_BYTES, self.storage_alignment)?;
                // the program, then its descriptors, each a run as fast as
                // copying, a chain of the two goes a word at a time
                let (program, descriptors) = self.ring(offset, PROGRAM_BYTES).as_chunks_mut::<4>().0.split_at_mut(unit.program.len());
                for (out, word) in program.iter_mut().zip(unit.program.iter()) {
                    *out = word.to_le_bytes();
                }
                for (out, word) in descriptors.iter_mut().zip(unit.descriptors.iter()) {
                    *out = word.to_le_bytes();
                }
                self.programs.insert(fingerprint, offset);
                offset
            }
        };

        // only the input registers the program reads are worked out, the
        // others read zero. for each, the byte its attribute starts at in
        // the first vertex, the bytes from one vertex to the next, and
        // offset | type << 8 | count << 16, no count reading the default
        let read = unit.inputs_read();
        let mut reads = [0usize; INPUT_REGISTERS];
        let mut count = 0;
        for register in (0..INPUT_REGISTERS).filter(|r| read & (1 << r) != 0) {
            reads[count] = register;
            count += 1;
        }
        let registers = &reads[..count];
        let mut attributes = [[0u32; 4]; INPUT_REGISTERS];
        let mut defaults = [crate::shader::ZERO; INPUT_REGISTERS];
        let mut starts = [0u32; 12];
        let input_bytes = match shading.inputs {
            Inputs::Decoded(decoded) => {
                // the registers a vertex reads in turn, four floats each
                let stride = registers.len() as u32 * 16;
                for (slot, &register) in registers.iter().enumerate() {
                    attributes[register] = [0, stride, (slot as u32 * 16) | 3 << 8 | 4 << 16, 0];
                }
                ((decoded.len() * registers.len()).max(1) * 16) as u64
            }
            Inputs::Raw(raw) => {
                // each array from a word on, and a word after the last, a
                // value can run into the word after its own
                let mut total = 0;
                for (start, &(_, len)) in starts.iter_mut().zip(&raw.arrays) {
                    *start = total;
                    total += len.div_ceil(4) * 4;
                }
                for &register in registers {
                    defaults[register] = raw.defaults[register];
                    if let Some(field) = raw.fields[register] {
                        let format = field.offset | field.ty << 8 | field.count << 16;
                        attributes[register] = [starts[field.array], field.stride, format, 0];
                    }
                }
                total as u64 + 4
            }
        };
        // the arrays go in as slices, a flattened iterator has no length to
        // reserve by and pushes a word at a time
        let mut block = std::mem::take(&mut self.shading);
        let words = &mut block.words;
        words.clear();
        words.extend(unit.float_uniforms.as_flattened().iter().map(|value| value.to_bits()));
        for [count, start, step, _] in unit.int_uniforms {
            words.extend([count as u32, start as u32, step as i8 as u32, 0]);
        }
        words.extend([unit.bool_uniforms as u32, unit.entry_point, 0, 0]);
        words.extend(shading.semantics);
        words.extend([depth_map.scale.to_bits(), depth_map.offset.to_bits(), 0, 0]);
        let (x, y, width, height) = shading.viewport;
        words.extend([x, y, width, height].map(f32::to_bits));
        words.extend_from_slice(attributes.as_flattened());
        words.extend(defaults.as_flattened().iter().map(|value| value.to_bits()));
        let uniforms = self.stage_block(&mut block, SHADING_SIZE, self.storage_alignment);
        self.shading = block;
        let uniforms = uniforms?;

        let inputs = self.stage(input_bytes, self.storage_alignment)?;
        match shading.inputs {
            Inputs::Decoded(decoded) => {
                let staging = self.ring(inputs, input_bytes);
                // the inner loop copies a register at a time, simple enough to
                // run at memory speed, a vertex can have thousands of them
                let mut out = staging.as_chunks_mut::<16>().0.iter_mut();
                for input in decoded {
                    for &register in registers {
                        let Some(slot) = out.next() else { break };
                        let [x, y, z, w] = input[register].map(f32::to_le_bytes);
                        *slot = [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3], w[0], w[1], w[2], w[3]];
                    }
                }
            }
            Inputs::Raw(raw) => {
                for (&(addr, len), &start) in raw.arrays.iter().zip(&starts) {
                    let bytes = memory.slice(addr, len as usize).ok_or("a vertex array is not in one piece")?;
                    self.ring(inputs + start as u64, len as u64).copy_from_slice(bytes);
                }
            }
        }

        let index_bytes = (shading.indices.len() * 4) as u64;
        let indices = self.stage(index_bytes, 4)?;
        for (out, index) in self.ring(indices, index_bytes).as_chunks_mut::<4>().0.iter_mut().zip(shading.indices) {
            *out = index.to_le_bytes();
        }
        Ok([(program, PROGRAM_BYTES), (uniforms, SHADING_SIZE), (inputs, input_bytes), (indices, index_bytes)])
    }

    pub(super) fn draw<M: GpuMemory>(&mut self, memory: &mut M, draw: &Draw) -> Result<(), String> {
        let [left, bottom, right, top] = draw.scissor;
        let (left, bottom) = (left.max(0), bottom.max(0));
        let (right, top) = (right.min(draw.width as i32), top.min(draw.height as i32));
        let empty = match draw.geometry {
            Geometry::Placed { indices, .. } => indices.is_empty(),
            Geometry::Shaded(shading) => shading.indices.is_empty(),
        };
        if empty || right <= left || top <= bottom {
            return Ok(());
        }
        // blocks used again add nothing to the ring, so a batch can run on
        // past where it used to be handed over, which moves when that
        // happens but not what is drawn
        if self.used > RING_FLUSH {
            self.submit()?;
        }
        self.begin()?;

        // the rows of memory the draw can touch, window rows run up from the
        // bottom of the buffer
        let rows = ((draw.height as i32 - top) as u32, (draw.height as i32 - bottom) as u32);
        // titles draw both screens into one buffer, the bottom one into the
        // start of the top one's memory. a taller buffer over the same memory
        // takes the draw, its first rows of memory are the window's top rows,
        // so the window goes up by the rows it lacks
        let height = self.taller(draw.target, Kind::Color(draw.format), draw.width, draw.height);
        let height = match draw.depth {
            Some((addr, bytes)) if self.taller(addr, Kind::Depth(bytes), draw.width, draw.height) != height => draw.height,
            _ => height,
        };
        let raise = (height - draw.height) as i32;
        // a flush looking one of them up drops what the other had checked
        let (color, depth) = loop {
            let size = (draw.width, height);
            let color = self.surface_rows(memory, draw.target, size, Kind::Color(draw.format), true, Some(rows))?;
            let depth = match draw.depth {
                Some((addr, bytes)) => Some(self.surface_rows(memory, addr, size, Kind::Depth(bytes), true, Some(rows))?),
                None => None,
            };
            if self.surfaces[color].checked && depth.is_none_or(|d| self.surfaces[d].checked) {
                break (color, depth);
            }
        };
        let (bottom, top) = (bottom + raise, top + raise);

        let mut views = [self.blank.view; 3];
        let mut samplers = [vk::Sampler::null(); 3];
        let mut enabled = 0;
        for (unit, bound) in draw.textures.iter().enumerate() {
            match bound {
                Some(bound) => {
                    views[unit] = match bound.drawn {
                        Some(drawn) => self.copy_texture(&drawn)?,
                        None => self.texture(bound)?,
                    };
                    samplers[unit] = self.sampler(bound.linear, bound.wrap_s, bound.wrap_t)?;
                    enabled |= 1 << unit;
                }
                None => samplers[unit] = self.sampler(false, Wrap::ClampToEdge, Wrap::ClampToEdge)?,
            }
        }
        let tables = match draw.lighting.is_some() || draw.proctex || draw.fog {
            true => self.tables(draw.tables, draw.proctex_tables, draw.fog_table)?,
            false => 0,
        };

        let (width, height) = (draw.width as f32, draw.height as f32);
        let depth_map = draw.depth_map;
        self.collect();
        let (vertex_count, (vertex_offset, index_offset), shaded, (vertex, translating)) = match draw.geometry {
            Geometry::Placed { vertices, indices } => {
                let offsets = self.place(vertices, indices, (width, height), depth_map)?;
                (indices.len(), offsets, None, (VertexStage::Placed, false))
            }
            Geometry::Shaded(shading) => {
                let staged = self.stage_shading(memory, shading, depth_map)?;
                (shading.indices.len(), (0, 0), Some(staged), self.vertex_stage(shading))
            }
        };


        let r = draw.registers;
        let mut block = std::mem::take(&mut self.uniforms);
        let words = &mut block.words;
        words.clear();
        for base in STAGE_REGISTERS {
            words.extend([r[base], r[base + 1], r[base + 2], r[base + 3], r[base + 4], 0, 0, 0]);
        }
        let texture_config = (r[REG_TEXTURE_CONFIG] & !7) | enabled;
        words.extend([r[REG_UPDATE_BUFFER], r[REG_BUFFER_COLOR], r[REG_ALPHA_TEST], texture_config]);
        for base in TEXTURE_UNIT_BASES {
            words.extend([r[base + 2], r[base], 0, 0]);
        }
        let depth_flags = depth_map.w_buffer as u32 | (shaded.is_some() as u32) << 1;
        // the depth map goes through the viewport where it fits, and what
        // the CPU placed carries its depth in its position, so the fragment
        // shader leaves depth alone and the GPU skips what is hidden before
        // shading it. a w-buffer is not linear on the screen, the fragment
        // shader works that out as before
        let mapped = (depth_map.offset, depth_map.offset - depth_map.scale);
        let (writes_depth, depth_range) = match draw.geometry {
            _ if depth_map.w_buffer => (true, (0.0, 1.0)),
            Geometry::Placed { .. } => (false, (0.0, 1.0)),
            Geometry::Shaded(_) if [mapped.0, mapped.1].iter().all(|d| (0.0..=1.0).contains(d)) => (false, mapped),
            Geometry::Shaded(_) => (true, (0.0, 1.0)),
        };
        words.extend([depth_flags, draw.lighting.is_some() as u32, depth_map.scale.to_bits(), depth_map.offset.to_bits()]);
        match draw.lighting {
            Some(lighting) => lighting.pack(words),
            None => words.resize(UNIFORM_WORDS - PROCTEX_WORDS - FOG_WORDS, 0),
        }
        let procedural = [
            proctex::REG_CONFIG,
            proctex::REG_CONFIG + 1,
            proctex::REG_CONFIG + 2,
            proctex::REG_CONFIG + 3,
            proctex::REG_CONFIG + 4,
            proctex::REG_CONFIG + 5,
        ];
        words.extend(procedural.map(|register| r[register]));
        words.extend([0, 0]);
        words.extend([r[fog::REG_COLOR], 0, 0, 0]);
        let uniform_offset = self.stage_block(&mut block, UNIFORM_SIZE, self.uniform_alignment);
        self.uniforms = block;
        let uniform_offset = uniform_offset?;

        // what the draw may change
        let color_mask = if r[REG_COLOR_BUFFER_WRITE] != 0 { (r[REG_DEPTH_COLOR_MASK] >> 8) & 0xF } else { 0 };
        let mask = r[REG_DEPTH_COLOR_MASK];
        let writable = r[REG_DEPTH_STENCIL_WRITE] != 0;
        let depth_test = mask & 1 != 0;
        let depth_write = writable && mask & (1 << 12) != 0;
        let stencil_test = draw.depth.is_some_and(|(_, bytes)| bytes == 4) && r[REG_STENCIL_TEST] & 1 != 0;
        // only the bits compile_pipeline reads, so draws that differ
        // elsewhere share a pipeline
        let blend = (r[REG_COLOR_OPERATION] & 0x100 != 0).then_some(r[REG_BLEND_FUNC] & 0xFFFF_0707);
        let mut logic_op = LogicOp::read(r);
        let mut color_mask = color_mask;
        if !self.logic_ops {
            // without logic ops, a draw that keeps the colors can still just
            // not write them
            if logic_op == Some(LogicOp::Noop) {
                color_mask = 0;
            }
            logic_op = None;
        }
        // only the bits the shader reads, so draws that differ elsewhere
        // share a pipeline
        let mut fragment = [0u32; FRAGMENT_CONSTANTS];
        for (i, base) in STAGE_REGISTERS.into_iter().enumerate() {
            fragment[i] = r[base] & 0x0FFF_0FFF;
            fragment[6 + i] = r[base + 1] & 0x0077_7FFF;
            fragment[12 + i] = r[base + 2] & 0x000F_000F;
            fragment[18 + i] = r[base + 4] & 0x0003_0003;
        }
        fragment[24] = r[REG_UPDATE_BUFFER] & 0xFF00;
        fragment[25] = texture_config & 0x2707;
        fragment[26] = draw.lighting.is_some() as u32;
        fragment[27] = r[REG_ALPHA_TEST] & 0x71;
        fragment[28] = depth_flags | ((writes_depth as u32) * WRITES_DEPTH);
        // what stands in for a translated program's pipeline, the
        // interpreter, has the program staged and bound all the same. what
        // is dynamic is set below rather than held by the pipeline
        let dynamic = self.dynamic;
        let key = PipelineKey {
            blend: blend.filter(|_| !dynamic.blend),
            logic_op: logic_op.filter(|_| !dynamic.logic_op),
            mask: if dynamic.blend { 0xF } else { color_mask },
            depth: depth.is_some(),
            vertex,
            fragment,
        };
        let pipeline = self.pipeline_for(key, translating)?;

        self.mark(Work::Draw, false);
        if self.uploads {
            self.end_rendering();
            self.barrier();
            self.uploads = false;
        }
        if self.rendering != Some((color, depth)) {
            self.end_rendering();
            self.barrier();
            self.begin_rendering(color, depth);
        }
        if color_mask != 0 {
            self.surfaces[color].drew(rows);
            self.surfaces[color].guard_writes(memory);
            self.overdrawn(color, rows, depth);
        }
        if let Some(depth) = depth {
            if depth_write || (stencil_test && writable) {
                self.surfaces[depth].drew(rows);
                self.overdrawn(depth, rows, Some(color));
                // only depth, which titles read to tell what is in view.
                // Super Mario 3D Land reads small color buffers back on
                // loading a course and stalls on what the GPU drew there,
                // it goes on with them as memory holds them
                self.surfaces[depth].guard(memory);
                self.surfaces[depth].guard_writes(memory);
            }
        }

        let image_infos: [[vk::DescriptorImageInfo; 1]; 3] = std::array::from_fn(|unit| {
            [vk::DescriptorImageInfo::default()
                .sampler(samplers[unit])
                .image_view(views[unit])
                .image_layout(vk::ImageLayout::GENERAL)]
        });
        let uniform_info = [vk::DescriptorBufferInfo::default().buffer(self.ring.buffer).offset(uniform_offset).range(UNIFORM_SIZE)];
        let tables_info = [vk::DescriptorBufferInfo::default().buffer(self.ring.buffer).offset(tables).range(TABLES_SIZE)];
        let vertex_infos = shaded.map(|staged| {
            staged.map(|(offset, range)| [vk::DescriptorBufferInfo::default().buffer(self.ring.buffer).offset(offset).range(range)])
        });
        let mut writes = [vk::WriteDescriptorSet::default(); 8];
        let fixed = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos[0]),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos[1]),
            vk::WriteDescriptorSet::default()
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos[2]),
            vk::WriteDescriptorSet::default()
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_info),
            vk::WriteDescriptorSet::default()
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&tables_info),
        ];
        writes[..fixed.len()].copy_from_slice(&fixed);
        let mut count = fixed.len();
        if let Some(infos) = &vertex_infos {
            for (binding, info) in (5..).zip(&infos[..3]) {
                writes[count] = vk::WriteDescriptorSet::default()
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info);
                count += 1;
            }
        }
        let writes = &writes[..count];
        // where the PICA's viewport puts clip space, for what the GPU
        // shades, the whole target for what the CPU placed on it
        let (x, y, viewport_width, viewport_height) = match draw.geometry {
            Geometry::Shaded(shading) => shading.viewport,
            Geometry::Placed { .. } => (0.0, 0.0, width, height),
        };
        let y = y + raise as f32;
        let cull = match draw.geometry {
            Geometry::Shaded(shading) if shading.cull == 1 => vk::CullModeFlags::FRONT,
            Geometry::Shaded(shading) if shading.cull != 0 => vk::CullModeFlags::BACK,
            _ => vk::CullModeFlags::NONE,
        };

        let commands = self.commands;
        let scale = self.scale;
        let viewport = vk::Viewport {
            x: x * scale as f32,
            y: y * scale as f32,
            width: viewport_width * scale as f32,
            height: viewport_height * scale as f32,
            min_depth: depth_range.0,
            max_depth: depth_range.1,
        };
        let face = vk::StencilFaceFlags::FRONT_AND_BACK;
        let test = r[REG_STENCIL_TEST];
        let op = r[REG_STENCIL_OP];
        let stencil_op = |raw: u32| vk::StencilOp::from_raw((raw & 7) as i32);
        let constant = r[REG_BLEND_COLOR].to_le_bytes().map(|c| c as f32 / 255.0);
        // SAFETY: recording inside rendering, with everything the draw
        // reads staged in the ring and every image in the general layout
        unsafe {
            let device = &self.device;
            device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(commands, 0, &[viewport]);
            device.cmd_set_cull_mode(commands, cull);
            device.cmd_set_scissor(
                commands,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: left * scale as i32, y: bottom * scale as i32 },
                    extent: vk::Extent2D { width: (right - left) as u32 * scale, height: (top - bottom) as u32 * scale },
                }],
            );
            // with the test off the PICA still writes depth, which Vulkan
            // only does while testing
            device.cmd_set_depth_test_enable(commands, depth.is_some() && (depth_test || depth_write));
            device.cmd_set_depth_compare_op(
                commands,
                if depth_test { COMPARES[((mask >> 4) & 7) as usize] } else { vk::CompareOp::ALWAYS },
            );
            device.cmd_set_depth_write_enable(commands, depth.is_some() && depth_write);
            device.cmd_set_stencil_test_enable(commands, stencil_test);
            device.cmd_set_stencil_op(
                commands,
                face,
                stencil_op(op),
                stencil_op(op >> 8),
                stencil_op(op >> 4),
                COMPARES[((test >> 4) & 7) as usize],
            );
            device.cmd_set_stencil_compare_mask(commands, face, (test >> 24) & 0xFF);
            device.cmd_set_stencil_write_mask(commands, face, if writable { (test >> 8) & 0xFF } else { 0 });
            device.cmd_set_stencil_reference(commands, face, (test >> 16) & 0xFF);
            device.cmd_set_blend_constants(commands, &constant);
            if let Some(state) = &self.blend_state {
                state.cmd_set_color_blend_enable(commands, 0, &[blend.is_some().into()]);
                state.cmd_set_color_blend_equation(commands, 0, &[blend_equation(blend.unwrap_or(0))]);
                state.cmd_set_color_write_mask(commands, 0, &[vk::ColorComponentFlags::from_raw(color_mask)]);
                if let Some(ops) = &self.logic_op_state {
                    state.cmd_set_logic_op_enable(commands, logic_op.is_some());
                    ops.cmd_set_logic_op(commands, logic_op.map_or(vk::LogicOp::COPY, self::logic_op));
                }
            }
            self.push.cmd_push_descriptor_set(commands, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, writes);
            self.draws += 1;
            match vertex_infos {
                Some(infos) => {
                    device.cmd_bind_index_buffer(commands, self.ring.buffer, infos[3][0].offset, vk::IndexType::UINT32);
                    device.cmd_draw_indexed(commands, vertex_count as u32, 1, 0, 0, 0);
                }
                None => {
                    device.cmd_bind_vertex_buffers(commands, 0, &[self.ring.buffer], &[vertex_offset]);
                    device.cmd_bind_index_buffer(commands, self.ring.buffer, index_offset, vk::IndexType::UINT32);
                    device.cmd_draw_indexed(commands, vertex_count as u32, 1, 0, 0, 0);
                }
            }
        }
        Ok(())
    }

    fn begin_rendering(&mut self, color: usize, depth: Option<usize>) {
        self.renderings += 1;
        let surface = &self.surfaces[color];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: surface.width * self.scale, height: surface.height * self.scale },
        };
        let attachment = |view: vk::ImageView| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        };
        let colors = [attachment(surface.image.view)];
        let depth_attachment = depth.map(|d| attachment(self.surfaces[d].image.view));
        let mut info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors);
        if let Some(depth_attachment) = &depth_attachment {
            info = info.depth_attachment(depth_attachment).stencil_attachment(depth_attachment);
        }
        // SAFETY: recording, outside rendering, with images in the general
        // layout
        self.unfenced = true;
        unsafe { self.device.cmd_begin_rendering(self.commands, &info) };
        self.rendering = Some((color, depth));
    }

    /// runs what the batch recorded and writes everything it drew back to
    /// guest memory.
    pub(crate) fn flush<M: GpuMemory>(&mut self, memory: &mut M) -> Result<(), String> {
        let dirty = (0..self.surfaces.len()).filter(|&i| self.surfaces[i].dirty.is_some()).collect();
        self.write_back(memory, dirty)?;
        self.submit()?;
        self.wait()
    }

    /// makes guest memory right over a range something other than a draw
    /// is about to read or write, what the GPU drew there comes down, and
    /// the next draw looks at the memory again.
    pub(crate) fn sync<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        self.sync_where(memory, addr, len, |_| true)
    }

    /// the same over the depth buffers alone, for the CPU reading where a
    /// depth buffer was drawn. the color buffers there it reads as memory
    /// holds them, see draw, even over memory a depth buffer had before.
    pub(crate) fn sync_depth<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        self.sync_where(memory, addr, len, |surface| matches!(surface.kind, Kind::Depth(_)))
    }

    fn sync_where<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        len: u32,
        kept: impl Fn(&Surface) -> bool,
    ) -> Result<(), String> {
        let overlapping: Vec<usize> =
            (0..self.surfaces.len()).filter(|&i| kept(&self.surfaces[i]) && self.surfaces[i].overlaps(addr, len)).collect();
        let dirty: Vec<usize> = overlapping.iter().copied().filter(|&i| self.surfaces[i].dirty_overlaps(addr, len)).collect();
        self.write_back(memory, dirty)?;
        for i in overlapping {
            self.surfaces[i].checked = false;
        }
        Ok(())
    }

    /// a display transfer out of a buffer the GPU holds, one it drew or an
    /// earlier transfer wrote, done on the GPU, false when the input is
    /// anything else, for the CPU to do. the output stays on the GPU, and a
    /// copy of it heads for the host in the same batch, which goes to the
    /// GPU right away without waiting for it.
    pub(crate) fn display_transfer<M: GpuMemory>(&mut self, memory: &mut M, transfer: &Transfer) -> Result<bool, String> {
        let t = transfer;
        let input_kind = Kind::Color(t.input_format);
        let output_kind = Kind::Color(t.output_format);
        let output_size = (t.output_width, t.output_height);
        // a tiled output of part of a tile goes to the CPU, which leaves out
        // the pixels the layout puts past its end, as draws do
        let whole = t.output_width.is_multiple_of(8) && t.output_height.is_multiple_of(8);
        if t.copy.0 == 0 || t.copy.1 == 0 || (t.output_tiled && !whole) {
            return Ok(false);
        }
        // the input rows read, which can start some rows into a buffer the
        // GPU holds, titles draw both screens into one, a tiled buffer a
        // row of tiles at a time
        let tiled = !t.input_linear;
        let rows = t.copy.1 * t.scale.1;
        let first = if t.flip { t.input_height - rows } else { 0 };
        let step = if tiled { 8 } else { 1 };
        let step_bytes = step * t.input_width * input_kind.bytes();
        let row_in = |s: &Surface| {
            let offset = t.input.checked_sub(s.addr).filter(|offset| offset % step_bytes == 0)? / step_bytes * step;
            (offset + first + rows <= s.height).then_some(offset)
        };
        let Some((addr, input_size, row)) = self
            .surfaces
            .iter()
            .filter(|s| s.kind == input_kind && s.tiled == tiled && s.width == t.input_width)
            .filter_map(|s| Some((s, row_in(s)?)))
            .max_by_key(|&(s, row)| newest(s, (row + first, row + first + rows)))
            .map(|(s, row)| (s.addr, (s.width, s.height), row))
        else {
            return Ok(false);
        };
        let input_end = addr as u64 + (input_size.0 * input_size.1 * input_kind.bytes()) as u64;
        let output_end = t.output as u64 + (t.output_width * t.output_height * output_kind.bytes()) as u64;
        if (addr as u64) < output_end && (t.output as u64) < input_end {
            return Ok(false);
        }
        self.begin()?;
        // a flush looking one of them up drops what the other had checked
        let (source, target) = loop {
            // the rows it reads, what is drawn over the rest stays on the GPU
            let source = self.surface_rows(memory, addr, input_size, input_kind, tiled, Some((row + first, row + first + rows)))?;
            let target = self.surface(memory, t.output, output_size, output_kind, t.output_tiled)?;
            if self.surfaces[source].checked && self.surfaces[target].checked {
                break (source, target);
            }
        };
        // drawn scaled, every size and row is that many times more, and the
        // pixels averaged stay as many
        let n = self.scale as i32;
        let constants: [i32; 11] = [
            t.copy.0 as i32 * n,
            t.copy.1 as i32 * n,
            t.input_height as i32 * n,
            t.output_height as i32 * n,
            t.scale.0 as i32,
            t.scale.1 as i32,
            t.flip as i32,
            format_index(t.input_format),
            format_index(t.output_format),
            row as i32 * n,
            input_size.1 as i32 * n,
        ];
        let views = (self.surfaces[source].image.view, self.surfaces[target].image.view);
        self.dispatch_transfer(views, constants, (t.copy.0 * self.scale, t.copy.1 * self.scale))?;
        self.surfaces[target].changed();
        self.surfaces[target].guard_writes(memory);
        self.overdrawn(target, (0, self.surfaces[target].height), None);
        self.capture(target)?;
        // turned upright for showing at any scale, at the console's own too
        // that beats the CPU waiting for the GPU and decoding the buffer
        self.capture_screen(target)?;
        self.submit()?;
        Ok(true)
    }

    /// records the transfer shader going over size pixels, from one image
    /// to another, as the constants say.
    fn dispatch_transfer(&mut self, (source, target): (vk::ImageView, vk::ImageView), constants: [i32; 11], size: (u32, u32)) -> Result<(), String> {
        self.mark(Work::Transfer, false);
        let (layout, pipeline) = self.transfer_pipeline()?;
        self.end_rendering();
        self.barrier();
        let infos = [source, target].map(|view| [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)]);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&infos[0]),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&infos[1]),
        ];
        let mut bytes = [0u8; 44];
        for (bytes, constant) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(constants) {
            *bytes = constant.to_le_bytes();
        }
        // SAFETY: recording, outside rendering, on images in the general
        // layout made for storage
        unsafe {
            self.device.cmd_bind_pipeline(self.commands, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(self.commands, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            self.device.cmd_push_constants(self.commands, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            self.unfenced = true;
            self.device.cmd_dispatch(self.commands, size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
        Ok(())
    }

    /// the surface a texture is rows of, and the row it starts at, when
    /// they line up the way the PICA lays both out.
    /// a color buffer in its own format, or a depth buffer whose samples
    /// are as wide as the texture's pixels, d24s8 read as rgba8 or d24 as
    /// rgb8.
    fn texture_source(&self, texture: &DrawnTexture) -> Option<(usize, u32)> {
        let depth = match texture.format {
            ColorFormat::Rgba8 => Some(Kind::Depth(4)),
            ColorFormat::Rgb8 => Some(Kind::Depth(3)),
            _ => None,
        };
        let kinds = [Some(Kind::Color(texture.format)), depth];
        let tile_rows = 8 * texture.width * Kind::Color(texture.format).bytes();
        (0..self.surfaces.len())
            .filter_map(|i| {
                let s = &self.surfaces[i];
                if !(kinds.contains(&Some(s.kind)) && s.tiled && s.width == texture.width) {
                    return None;
                }
                let row = texture.addr.checked_sub(s.addr).filter(|offset| offset % tile_rows == 0)? / tile_rows * 8;
                (row + texture.height <= s.height).then_some((i, row))
            })
            .max_by_key(|&(i, row)| newest(&self.surfaces[i], (row, row + texture.height)))
    }

    /// whether a texture is rows of a surface the GPU drew and guest memory
    /// has not got back, which draw then copies on the GPU. rows a fill
    /// cleared of it hold what memory held then, when memory may have
    /// changed there since, or another surface drew over them, the texture
    /// comes from memory once what was drawn is written back.
    pub(crate) fn holds(&self, texture: &DrawnTexture) -> bool {
        let Some((index, row)) = self.texture_source(texture) else { return false };
        let surface = &self.surfaces[index];
        let changed = surface.cleared.is_some_and(|(from, to)| {
            let (from, to) = (from.max(row), to.min(row + texture.height));
            let at = surface.addr + from * surface.row_bytes();
            from < to
                && (!surface.checked
                    || (0..self.surfaces.len()).any(|i| i != index && self.surfaces[i].dirty_overlaps(at, (to - from) * surface.row_bytes())))
        });
        surface.dirty.is_some() && !changed
    }

    /// the image of a texture copied from the surface it is part of, copied
    /// again whenever the surface changed.
    fn copy_texture(&mut self, texture: &DrawnTexture) -> Result<vk::ImageView, String> {
        self.mark(Work::Copy, false);
        let (source, row) = self.texture_source(texture).ok_or("the buffer a texture was drawn into is gone")?;
        let generation = self.surfaces[source].generation;
        let batch = self.batch;
        if let Some(copy) = self.copies.get_mut(texture) {
            if copy.surface == source && copy.generation == generation {
                copy.used = batch;
                return Ok(copy.image.view);
            }
        }
        if !self.copies.contains_key(texture) {
            let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE;
            let image = self.image(texture.width * self.scale, texture.height * self.scale, COLOR_FORMAT, usage, vk::ImageAspectFlags::COLOR)?;
            self.copies.insert(*texture, Copied { image, surface: source, generation, used: batch });
        }
        let view = self.copies[texture].image.view;
        if let Kind::Depth(bytes) = self.surfaces[source].kind {
            self.copy_depth(source, row, bytes, view, texture)?;
        } else {
            // the texture's rows run top first, the surface's bottom first,
            // so the copy flips them, and its pixels round to the format as
            // memory would have them
            let format = format_index(texture.format);
            let n = self.scale;
            let (width, height) = ((texture.width * n) as i32, (texture.height * n) as i32);
            let (row, source_height) = ((row * n) as i32, (self.surfaces[source].height * n) as i32);
            let constants = [width, height, height, height, 1, 1, 1, format, format, row, source_height];
            self.dispatch_transfer((self.surfaces[source].image.view, view), constants, (texture.width * n, texture.height * n))?;
        }
        self.uploads = true;
        if let Some(copy) = self.copies.get_mut(texture) {
            copy.surface = source;
            copy.generation = generation;
            copy.used = batch;
        }
        Ok(view)
    }

    /// records a texture read out of a depth surface, its depth and stencil
    /// copied into a buffer and then made into the colors their bytes read
    /// as.
    fn copy_depth(&mut self, source: usize, row: u32, bytes: u32, view: vk::ImageView, texture: &DrawnTexture) -> Result<(), String> {
        let n = self.scale;
        let (width, height) = (self.surfaces[source].width * n, self.surfaces[source].height * n);
        let pixels = (width * height) as u64;
        let samples = self.samples(pixels * 5)?;
        let (layout, pipeline) = self.depth_pipeline()?;
        self.end_rendering();
        self.barrier();
        let extent = vk::Extent3D { width, height, depth: 1 };
        let region = |offset: u64, aspect: vk::ImageAspectFlags| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1))
                .image_extent(extent)
        };
        let regions = [region(0, vk::ImageAspectFlags::DEPTH), region(pixels * 4, vk::ImageAspectFlags::STENCIL)];
        let depths = [vk::DescriptorBufferInfo::default().buffer(samples).offset(0).range(pixels * 4)];
        let stencils = [vk::DescriptorBufferInfo::default().buffer(samples).offset(pixels * 4).range(pixels.next_multiple_of(4))];
        let target = [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&depths),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&stencils),
            vk::WriteDescriptorSet::default()
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&target),
        ];
        let (texture_width, texture_height) = (texture.width * n, texture.height * n);
        let constants = [texture_width as i32, texture_height as i32, width as i32, height as i32, (row * n) as i32, (bytes == 4) as i32];
        let constants: Vec<u8> = constants.iter().flat_map(|c| c.to_le_bytes()).collect();
        // SAFETY: recording, outside rendering, from a depth image in the
        // general layout into a buffer big enough for both aspects, then a
        // barrier before the shader reads it
        unsafe {
            let image = self.surfaces[source].image.image;
            self.unfenced = true;
            self.device.cmd_copy_image_to_buffer(self.commands, image, vk::ImageLayout::GENERAL, samples, &regions);
            self.barrier();
            self.device.cmd_bind_pipeline(self.commands, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(self.commands, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            self.device.cmd_push_constants(self.commands, layout, vk::ShaderStageFlags::COMPUTE, 0, &constants);
            self.unfenced = true;
            self.device.cmd_dispatch(self.commands, texture_width.div_ceil(8), texture_height.div_ceil(8), 1);
        }
        Ok(())
    }

    /// records a copy of a surface for the host to read later.
    fn capture(&mut self, index: usize) -> Result<(), String> {
        self.mark(Work::Capture, false);
        let (width, height) = (self.surfaces[index].width, self.surfaces[index].height);
        let size = (width * height * 4) as u64;
        if self.surfaces[index].capture.is_none() {
            let buffer = self.buffer(size, vk::BufferUsageFlags::TRANSFER_DST, true)?;
            self.surfaces[index].capture = Some(Capture { buffer, batch: 0, current: false, watched: false, pictures: VecDeque::new() });
        }
        self.end_rendering();
        self.barrier();
        if self.scale > 1 {
            self.blit(index, false);
            self.barrier();
        }
        let batch = self.batch;
        let surface = &mut self.surfaces[index];
        let source = surface.native.as_ref().unwrap_or(&surface.image).image;
        let Some(capture) = surface.capture.as_mut() else { unreachable!("made above") };
        let region = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width, height, depth: 1 })];
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: recording, outside rendering, into a buffer the size of
        // the image, which the host reads only once this batch is done
        unsafe {
            self.unfenced = true;
            self.device.cmd_copy_image_to_buffer(self.commands, source, vk::ImageLayout::GENERAL, capture.buffer.buffer, &region);
            self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&to_host));
        }
        capture.batch = batch;
        capture.current = true;
        Ok(())
    }

    /// records a copy of a scaled surface turned upright, for showing it.
    fn capture_screen(&mut self, index: usize) -> Result<(), String> {
        if self.direct {
            return self.upright_image(index);
        }
        self.mark(Work::Capture, false);
        // a row of the surface is a column of the screen
        let (width, height) = (self.surfaces[index].height * self.scale, self.surfaces[index].width * self.scale);
        let size = (width * height * 4) as u64;
        if self.surfaces[index].screen.is_none() {
            let buffer = self.buffer(size, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
            self.surfaces[index].screen = Some(Capture { buffer, batch: 0, current: false, watched: false, pictures: VecDeque::new() });
        }
        let (layout, pipeline) = self.upright_pipeline()?;
        self.end_rendering();
        self.barrier();
        let batch = self.batch;
        let surface = &mut self.surfaces[index];
        let Some(screen) = surface.screen.as_mut() else { unreachable!("made above") };
        let source = [vk::DescriptorImageInfo::default().image_view(surface.image.view).image_layout(vk::ImageLayout::GENERAL)];
        let pixels = [vk::DescriptorBufferInfo::default().buffer(screen.buffer.buffer).offset(0).range(size)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&source),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&pixels),
        ];
        let constants: Vec<u8> = [width as i32, height as i32].iter().flat_map(|c| c.to_le_bytes()).collect();
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: recording, outside rendering, from an image in the general
        // layout into a buffer the size of the screen, which the host reads
        // only once this batch is done
        unsafe {
            self.device.cmd_bind_pipeline(self.commands, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(self.commands, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            self.device.cmd_push_constants(self.commands, layout, vk::ShaderStageFlags::COMPUTE, 0, &constants);
            self.unfenced = true;
            self.device.cmd_dispatch(self.commands, width.div_ceil(8), height.div_ceil(8), 1);
            self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&to_host));
        }
        screen.batch = batch;
        screen.current = true;
        Ok(())
    }

    /// records a surface turned upright into the image a presenter sharing
    /// the device draws it from. the batch's first barrier waits for the
    /// presents submitted before, which may still read the image.
    fn upright_image(&mut self, index: usize) -> Result<(), String> {
        self.mark(Work::Capture, false);
        // a row of the surface is a column of the screen
        let (width, height) = (self.surfaces[index].height * self.scale, self.surfaces[index].width * self.scale);
        if self.surfaces[index].upright.is_none() {
            let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
            let image = self.image(width, height, COLOR_FORMAT, usage, vk::ImageAspectFlags::COLOR)?;
            self.surfaces[index].upright = Some(Upright { image, batch: 0, current: false });
        }
        let (layout, pipeline) = self.upright_image_pipeline()?;
        self.end_rendering();
        self.barrier();
        let batch = self.batch;
        let surface = &mut self.surfaces[index];
        let Some(upright) = surface.upright.as_mut() else { unreachable!("made above") };
        upright.batch = batch;
        upright.current = true;
        let source = [vk::DescriptorImageInfo::default().image_view(surface.image.view).image_layout(vk::ImageLayout::GENERAL)];
        let target = [vk::DescriptorImageInfo::default().image_view(upright.image.view).image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&source),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&target),
        ];
        let constants: Vec<u8> = [width as i32, height as i32].iter().flat_map(|c| c.to_le_bytes()).collect();
        // SAFETY: recording, outside rendering, between two images in the
        // general layout, the presenter reads the target only in a later
        // submission, behind a barrier of its own
        unsafe {
            self.device.cmd_bind_pipeline(self.commands, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(self.commands, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            self.device.cmd_push_constants(self.commands, layout, vk::ShaderStageFlags::COMPUTE, 0, &constants);
            self.unfenced = true;
            self.device.cmd_dispatch(self.commands, width.div_ceil(8), height.div_ceil(8), 1);
        }
        Ok(())
    }

    /// whether screens are shown straight from images on the GPU.
    pub(crate) fn direct(&self) -> bool {
        self.direct
    }

    /// the image a screen's picture is upright in, read back, and its width.
    #[cfg(test)]
    pub(crate) fn upright_pixels(&mut self, screen: ScreenRef) -> Result<(Vec<u8>, u32), String> {
        let (row_pixels, rows) = screen.size;
        let index = self.surfaces.iter().position(|s| s.addr == screen.addr && s.width == row_pixels && s.height == rows).ok_or("no surface")?;
        let image = self.surfaces[index].upright.as_ref().ok_or("nothing upright")?.image.image;
        let (width, height) = (rows * self.scale, row_pixels * self.scale);
        let buffer = self.buffer(u64::from(width * height * 4), vk::BufferUsageFlags::TRANSFER_DST, true)?;
        self.begin()?;
        self.end_rendering();
        self.unfenced = true;
        self.barrier();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width, height, depth: 1 });
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: recording, outside rendering, into a buffer the image fits
        unsafe {
            self.device.cmd_copy_image_to_buffer(self.commands, image, vk::ImageLayout::GENERAL, buffer.buffer, &[region]);
            self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&to_host));
        }
        self.submit()?;
        self.wait()?;
        // SAFETY: the GPU is done with the buffer, which is mapped and goes
        // right after
        unsafe {
            if buffer.incoherent {
                let range = [vk::MappedMemoryRange::default().memory(buffer.memory).offset(0).size(vk::WHOLE_SIZE)];
                self.device.invalidate_mapped_memory_ranges(&range).map_err(vk_error("invalidate memory"))?;
            }
            let pixels = std::slice::from_raw_parts(buffer.mapped, (width * height * 4) as usize).to_vec();
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
            Ok((pixels, width))
        }
    }

    /// the image a screen's picture is upright in, for a presenter sharing
    /// the device, and where the screen's pixels lie in it. none once a later
    /// picture took its image. the batch that drew it goes to the GPU first,
    /// so it runs before whatever the presenter submits next.
    pub(crate) fn screen_image(&mut self, screen: ScreenRef) -> Result<Option<crate::GpuScreen>, String> {
        if !self.direct {
            return Ok(None);
        }
        let (row_pixels, rows) = screen.size;
        let found = self.surfaces.iter().position(|s| {
            s.addr == screen.addr && s.width == row_pixels && s.height == rows && s.kind == Kind::Color(screen.format) && !s.tiled
        });
        let Some(index) = found else { return Ok(None) };
        let Some(upright) = self.surfaces[index].upright.as_ref().filter(|upright| upright.batch == screen.batch) else {
            return Ok(None);
        };
        let view = upright.image.view;
        if screen.batch >= self.batch {
            self.submit()?;
        }
        // the rows of the surface the screen shows are columns of the upright
        // picture, the pixels of each run down it from the bottom of a row
        let scale = self.scale;
        let (first, count) = screen.rows;
        let (width, height) = ((rows * scale) as f32, (row_pixels * scale) as f32);
        let (left, right) = ((first * scale) as f32, ((first + count) * scale) as f32);
        let (top, bottom) = (((row_pixels - screen.columns) * scale) as f32, (row_pixels * scale) as f32);
        Ok(Some(crate::GpuScreen {
            view,
            area: [left / width, top / height, (right - left) / width, (bottom - top) / height],
            bounds: [(left + 0.5) / width, (top + 0.5) / height, (right - 0.5) / width, (bottom - 0.5) / height],
        }))
    }

    /// the newest picture a display transfer left in a buffer for a screen,
    /// while guest memory still holds what the transfer wrote there, or the
    /// GPU has not written it back yet. from now on each picture it gets is
    /// kept as its batch finishes.
    pub(crate) fn screen(&mut self, addr: u32, (width, height): (u32, u32), stride: u32, format: ColorFormat, guest: &[u8]) -> Option<ScreenRef> {
        // the buffer the screen is rows of, maybe some rows into it
        let row_bytes = stride * Kind::Color(format).bytes();
        let found = self.surfaces.iter().position(|s| {
            s.kind == Kind::Color(format)
                && !s.tiled
                && s.width == stride
                && stride >= width
                && addr >= s.addr
                && (addr - s.addr).is_multiple_of(row_bytes)
                && (addr - s.addr) / row_bytes + height <= s.height
        })?;
        let s = &mut self.surfaces[found];
        let first = (addr - s.addr) / row_bytes;
        let shown = (first * row_bytes) as usize..((first + height) * row_bytes) as usize;
        if s.dirty.is_none() && !s.shadows(shown, guest) {
            return None;
        }
        let (surface, size) = (s.addr, (s.width, s.height));
        let batch = match &mut s.upright {
            Some(upright) => upright.current.then_some(upright.batch)?,
            None => {
                let screen = s.screen.as_mut().filter(|screen| screen.current)?;
                screen.watched = true;
                screen.batch
            }
        };
        Some(ScreenRef { addr: surface, size, format, batch, rows: (first, height), columns: width })
    }

    /// a screen's picture upright, RGBA the way the screen shows it, and the
    /// scale it is at, waiting for the GPU to finish it when it has not yet.
    /// none once a later picture took its buffer.
    pub(crate) fn picture(&mut self, screen: ScreenRef) -> Result<Option<Picture>, String> {
        // a presenter sharing the device has the pictures, the host none
        if self.direct {
            return Ok(None);
        }
        let (width, height) = screen.size;
        let find = |surfaces: &[Surface]| {
            surfaces.iter().position(|s| {
                s.addr == screen.addr && s.width == width && s.height == height && s.kind == Kind::Color(screen.format) && !s.tiled
            })
        };
        let picture = |surfaces: &[Surface], index: usize| {
            let capture = surfaces[index].screen.as_ref()?;
            capture.pictures.iter().find(|(batch, _)| *batch == screen.batch).map(|(_, image)| image.clone())
        };
        let Some(index) = find(&self.surfaces) else { return Ok(None) };
        let scale = self.scale;
        let crop = |image: Arc<Vec<u8>>| Some((crop(image, screen, scale), scale));
        if let Some(image) = picture(&self.surfaces, index) {
            return Ok(crop(image));
        }
        // the buffer holds it until a later picture is drawn over it
        if self.surfaces[index].screen.as_ref().is_none_or(|capture| capture.batch != screen.batch) {
            return Ok(None);
        }
        self.wait_for(screen.batch)?;
        if let Some(image) = picture(&self.surfaces, index) {
            return Ok(crop(image));
        }
        // its batch finished before a screen showed the surface
        self.read_picture(index)?;
        Ok(picture(&self.surfaces, index).and_then(crop))
    }

    /// reads the picture a finished batch left in a surface's screen buffer.
    fn read_picture(&mut self, index: usize) -> Result<(), String> {
        let Some(capture) = self.surfaces[index].screen.as_mut() else { return Ok(()) };
        let buffer = &capture.buffer;
        if buffer.incoherent {
            let range = [vk::MappedMemoryRange::default().memory(buffer.memory).offset(0).size(vk::WHOLE_SIZE)];
            // SAFETY: the memory is mapped and the GPU is done with it
            unsafe { self.device.invalidate_mapped_memory_ranges(&range) }.map_err(vk_error("invalidate memory"))?;
        }
        // SAFETY: the batch that filled the buffer is done, and nothing
        // writes it again before this returns
        let data = unsafe { std::slice::from_raw_parts(buffer.mapped, buffer.size as usize) };
        capture.pictures.push_back((capture.batch, Arc::new(data.to_vec())));
        while capture.pictures.len() > 2 {
            capture.pictures.pop_front();
        }
        Ok(())
    }

    /// the pipeline display transfers run on, made the first time.
    fn transfer_pipeline(&mut self) -> Result<(vk::PipelineLayout, vk::Pipeline), String> {
        if self.transfer.is_none() {
            let images = [vk::DescriptorType::STORAGE_IMAGE; 2];
            self.transfer = Some(self.compute(TRANSFER_SPIRV, &images, 11 * 4)?);
        }
        let transfer = self.transfer.as_ref().expect("made above");
        Ok((transfer.layout, transfer.pipeline))
    }

    /// the pipeline screens are turned upright into an image with, for a
    /// presenter sharing the device, made the first time.
    fn upright_image_pipeline(&mut self) -> Result<(vk::PipelineLayout, vk::Pipeline), String> {
        if self.upright_image.is_none() {
            let bindings = [vk::DescriptorType::STORAGE_IMAGE; 2];
            self.upright_image = Some(self.compute(UPRIGHT_IMAGE_SPIRV, &bindings, 2 * 4)?);
        }
        let upright = self.upright_image.as_ref().expect("made above");
        Ok((upright.layout, upright.pipeline))
    }

    /// the pipeline scaled screens are turned upright with, made the first
    /// time.
    fn upright_pipeline(&mut self) -> Result<(vk::PipelineLayout, vk::Pipeline), String> {
        if self.upright.is_none() {
            let bindings = [vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_BUFFER];
            self.upright = Some(self.compute(UPRIGHT_SPIRV, &bindings, 2 * 4)?);
        }
        let upright = self.upright.as_ref().expect("made above");
        Ok((upright.layout, upright.pipeline))
    }

    /// the pipeline textures are read out of depth buffers with, made the
    /// first time.
    fn depth_pipeline(&mut self) -> Result<(vk::PipelineLayout, vk::Pipeline), String> {
        if self.depth.is_none() {
            let bindings = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_IMAGE];
            self.depth = Some(self.compute(DEPTH_SPIRV, &bindings, 6 * 4)?);
        }
        let depth = self.depth.as_ref().expect("made above");
        Ok((depth.layout, depth.pipeline))
    }

    /// a compute pipeline running spirv, with a pushed descriptor per
    /// binding and push constants of push bytes.
    fn compute(&self, spirv: &[u8], bindings: &[vk::DescriptorType], push: u32) -> Result<Compute, String> {
        // SAFETY: plain object creation on our device, with create infos
        // that live as long as each call
        unsafe {
            let words = ash::util::read_spv(&mut Cursor::new(spirv)).map_err(|e| e.to_string())?;
            let shader = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vk_error("create a shader module"))?;
            let bindings: Vec<_> = bindings
                .iter()
                .enumerate()
                .map(|(binding, &kind)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding as u32)
                        .descriptor_type(kind)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .map_err(vk_error("create a descriptor set layout"))?;
            let set_layouts = [set_layout];
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push)];
            let layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&ranges),
                    None,
                )
                .map_err(vk_error("create a pipeline layout"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = self
                .device
                .create_compute_pipelines(self.pipeline_cache, &[info], None)
                .map_err(|(_, error)| format!("could not create a pipeline, {error}"))?[0];
            Ok(Compute { shader, set_layout, layout, pipeline })
        }
    }

    /// a buffer in the GPU's own memory of at least size bytes for depth
    /// samples on their way to a texture, grown when it has to be.
    fn samples(&mut self, size: u64) -> Result<vk::Buffer, String> {
        if let Some(samples) = self.samples.as_ref().filter(|samples| samples.size >= size) {
            return Ok(samples.buffer);
        }
        // the batch before may still use the old one
        self.submit()?;
        self.wait()?;
        self.begin()?;
        if let Some(old) = self.samples.take() {
            // SAFETY: the GPU is done with everything that used it
            unsafe {
                self.device.destroy_buffer(old.buffer, None);
                self.device.free_memory(old.memory, None);
            }
        }
        let size = size.next_power_of_two();
        // SAFETY: plain object creation on our device
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = self
                .device
                .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
                .map_err(vk_error("create a buffer"))?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let (memory, _) = self.allocate(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(vk_error("bind buffer memory"))?;
            self.samples = Some(Local { buffer, memory, size });
            Ok(buffer)
        }
    }

    /// hands the batch being recorded to the GPU without waiting for it.
    /// recording goes on in another command buffer, once the GPU is done
    /// with the batch that used it last.
    fn submit(&mut self) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        self.end_rendering();
        self.mark(Work::Other, true);
        if let Some(&pool) = self.timing.as_ref().and_then(|timing| timing.statistics.as_ref()?.get(&self.commands)) {
            // SAFETY: begun when the batch began, outside rendering
            unsafe { self.device.cmd_end_query(self.commands, pool, 0) };
        }
        self.retire_finished()?;
        if self.free.is_empty() {
            self.block(Wait::Room)?;
        }
        // SAFETY: the command buffer is recording and gets submitted once
        unsafe {
            self.device.end_command_buffer(self.commands).map_err(vk_error("end a command buffer"))?;
            let buffers = [self.commands];
            let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
            self.device.queue_submit(self.queue, &submit, self.fence).map_err(vk_error("submit"))?;
        }
        self.recording = false;
        let next = self.free.pop().expect("a frame was freed above");
        let submitted = Frame {
            commands: std::mem::replace(&mut self.commands, next.commands),
            fence: std::mem::replace(&mut self.fence, next.fence),
            ring: std::mem::replace(&mut self.ring, next.ring),
            pending: Some(self.batch),
        };
        self.in_flight.push_back(submitted);
        for surface in &mut self.surfaces {
            surface.checked = false;
        }
        self.used = 0;
        self.tables = None;
        self.programs.clear();
        self.uniforms.at = None;
        self.shading.at = None;
        self.batch += 1;

        // textures nobody drew with for a while go, long after the GPU
        // last used them
        let batch = self.batch;
        let stale: Vec<usize> = self.textures.iter().filter(|(_, t)| batch - t.used > 600).map(|(&k, _)| k).collect();
        for key in stale {
            if let Some(texture) = self.textures.remove(&key) {
                self.destroy_image(&texture.image);
            }
        }
        let stale: Vec<DrawnTexture> = self.copies.iter().filter(|(_, c)| batch - c.used > 600).map(|(&k, _)| k).collect();
        for key in stale {
            if let Some(copy) = self.copies.remove(&key) {
                self.destroy_image(&copy.image);
            }
        }
        Ok(())
    }

    /// waits for the GPU to finish everything handed to it.
    fn wait(&mut self) -> Result<(), String> {
        while !self.in_flight.is_empty() {
            self.block(Wait::Done)?;
        }
        Ok(())
    }

    /// waits for a batch, handing it to the GPU first if it is the one
    /// being recorded.
    fn wait_for(&mut self, batch: u64) -> Result<(), String> {
        if batch == self.batch {
            self.submit()?;
        }
        while self.in_flight.front().is_some_and(|frame| frame.pending.is_some_and(|pending| pending <= batch)) {
            self.block(Wait::Done)?;
        }
        Ok(())
    }

    /// retires the oldest batch in flight, timing the wait for it when the
    /// GPU's times are being measured.
    fn block(&mut self, why: Wait) -> Result<(), String> {
        let start = self.timing.is_some().then(std::time::Instant::now);
        self.retire_oldest()?;
        if let (Some(start), Some(timing)) = (start, self.timing.as_mut()) {
            let waited = &mut timing.waited[why as usize];
            waited.0 += start.elapsed().as_secs_f64() * 1e3;
            waited.1 += 1;
        }
        Ok(())
    }

    /// frees the batches the GPU already finished, oldest first.
    fn retire_finished(&mut self) -> Result<(), String> {
        while let Some(frame) = self.in_flight.front() {
            // SAFETY: a fence of ours, submitted with its batch
            let done = unsafe { self.device.get_fence_status(frame.fence) }.map_err(vk_error("ask the GPU"))?;
            if !done {
                break;
            }
            self.retire_oldest()?;
        }
        Ok(())
    }

    /// waits for the oldest batch in flight and frees what it used.
    fn retire_oldest(&mut self) -> Result<(), String> {
        let Some(mut frame) = self.in_flight.pop_front() else { return Ok(()) };
        // SAFETY: the fence belongs to the batch, and the command buffer is
        // reset only once the GPU is done with it
        unsafe {
            self.device.wait_for_fences(&[frame.fence], true, u64::MAX).map_err(vk_error("wait for the GPU"))?;
            self.device.reset_fences(&[frame.fence]).map_err(vk_error("reset a fence"))?;
            self.device
                .reset_command_buffer(frame.commands, vk::CommandBufferResetFlags::empty())
                .map_err(vk_error("reset a command buffer"))?;
        }
        let batch = frame.pending.take();
        self.add_times(frame.commands)?;
        self.free.push(frame);
        // pictures for screens are read out before a later batch can draw
        // over them
        for index in 0..self.surfaces.len() {
            let finished = self.surfaces[index].screen.as_ref().is_some_and(|capture| capture.watched && Some(capture.batch) == batch);
            if finished {
                self.read_picture(index)?;
            }
        }
        Ok(())
    }

    /// writes the given surfaces back to guest memory, from their captures
    /// when those still hold them, else running the batch and reading them
    /// back.
    fn write_back<M: GpuMemory>(&mut self, memory: &mut M, dirty: Vec<usize>) -> Result<(), String> {
        self.mark(Work::Download, false);
        let (captured, others): (Vec<usize>, Vec<usize>) =
            dirty.into_iter().partition(|&i| self.surfaces[i].captured().is_some());
        if !others.is_empty() {
            self.run(memory, others)?;
        }
        for i in captured {
            // running the others may have finished it already
            let Some(capture) = self.surfaces[i].captured() else { continue };
            let batch = capture.batch;
            self.wait_for(batch)?;
            let Some(capture) = self.surfaces[i].captured() else { continue };
            let buffer = &capture.buffer;
            if buffer.incoherent {
                let range = [vk::MappedMemoryRange::default().memory(buffer.memory).offset(0).size(vk::WHOLE_SIZE)];
                // SAFETY: the memory is mapped and the GPU is done with it
                unsafe { self.device.invalidate_mapped_memory_ranges(&range) }.map_err(vk_error("invalidate memory"))?;
            }
            // SAFETY: the batch that filled the buffer is done, and nothing
            // writes it again before this returns
            let data = unsafe { std::slice::from_raw_parts(buffer.mapped, buffer.size as usize) };
            self.store(memory, i, data);
        }
        Ok(())
    }

    /// runs what the batch recorded, then reads the given surfaces back and
    /// writes them to guest memory.
    fn run<M: GpuMemory>(&mut self, memory: &mut M, dirty: Vec<usize>) -> Result<(), String> {
        self.begin()?;
        self.end_rendering();

        // room for every surface to write back, colors as RGBA, depth as a
        // word and stencil as a byte per sample
        let size_of = |s: &Surface| {
            let pixels = (s.width * s.height) as u64;
            match s.kind {
                Kind::Color(_) => pixels * 4,
                Kind::Depth(_) => pixels * 5,
            }
        };
        let total: u64 = dirty.iter().map(|&i| size_of(&self.surfaces[i])).sum();
        if total > self.readback.as_ref().map_or(0, |b| b.size) {
            if let Some(old) = self.readback.take() {
                // SAFETY: the old buffer was last used by a batch already
                // waited for
                unsafe {
                    self.device.destroy_buffer(old.buffer, None);
                    self.device.free_memory(old.memory, None);
                }
            }
            self.readback = Some(self.buffer(total.next_power_of_two(), vk::BufferUsageFlags::TRANSFER_DST, true)?);
        }

        let mut offsets = Vec::with_capacity(dirty.len());
        self.barrier();
        if self.scale > 1 {
            for &i in &dirty {
                self.blit(i, false);
            }
            self.barrier();
        }
        let readback = self.readback.as_ref().map_or(vk::Buffer::null(), |b| b.buffer);
        let mut offset = 0;
        for &i in &dirty {
            let s = &self.surfaces[i];
            let extent = vk::Extent3D { width: s.width, height: s.height, depth: 1 };
            let region = |at: u64, aspect: vk::ImageAspectFlags| {
                vk::BufferImageCopy::default()
                    .buffer_offset(at)
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1))
                    .image_extent(extent)
            };
            let pixels = (s.width * s.height) as u64;
            let regions: Vec<_> = match s.kind {
                Kind::Color(_) => vec![region(offset, vk::ImageAspectFlags::COLOR)],
                Kind::Depth(_) => vec![
                    region(offset, vk::ImageAspectFlags::DEPTH),
                    region(offset + pixels * 4, vk::ImageAspectFlags::STENCIL),
                ],
            };
            let image = s.native.as_ref().unwrap_or(&s.image).image;
            // SAFETY: recording, outside rendering, into a buffer big enough
            // for every region
            self.unfenced = true;
            unsafe { self.device.cmd_copy_image_to_buffer(self.commands, image, vk::ImageLayout::GENERAL, readback, &regions) };
            offsets.push(offset);
            offset += size_of(s);
        }
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: recording
        unsafe { self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&to_host)) };

        self.submit()?;
        self.wait()?;
        let Some(readback) = self.readback.as_ref() else { return Ok(()) };
        if readback.incoherent {
            let range = [vk::MappedMemoryRange::default().memory(readback.memory).offset(0).size(vk::WHOLE_SIZE)];
            // SAFETY: the memory is mapped and the GPU is done with it
            unsafe { self.device.invalidate_mapped_memory_ranges(&range) }.map_err(vk_error("invalidate memory"))?;
        }
        let mapped = readback.mapped;
        for (&i, &offset) in dirty.iter().zip(&offsets) {
            // SAFETY: the GPU finished writing the buffer, and the range was
            // sized for this surface
            let data = unsafe { std::slice::from_raw_parts(mapped.add(offset as usize), size_of(&self.surfaces[i]) as usize) };
            self.store(memory, i, data);
        }
        Ok(())
    }

    /// writes what the GPU read back of a surface to guest memory.
    fn store<M: GpuMemory>(&mut self, memory: &mut M, index: usize, data: &[u8]) {
        // the drawing goes over some rows of a shadow a pixel stood in for
        self.surfaces[index].shadow();
        let s = &self.surfaces[index];
        let (width, height, kind, tiled) = (s.width, s.height, s.kind, s.tiled);
        let pixels = (width * height) as usize;
        // only the rows drawn, the others can be another buffer's by now
        let (first, last) = s.dirty.unwrap_or((0, height));
        let mut bytes = vec![0u8; pixels * kind.bytes() as usize];
        match kind {
            Kind::Color(format) => {
                let bpp = format.bytes_per_pixel();
                for y in first..last {
                    for x in 0..width {
                        let at = (((height - 1 - y) * width + x) * 4) as usize;
                        let rgba = [data[at], data[at + 1], data[at + 2], data[at + 3]];
                        let out = pixel_offset(tiled, x, y, width, bpp as u32);
                        format.encode(rgba, &mut bytes[out..out + bpp]);
                    }
                }
            }
            Kind::Depth(sample) => {
                let (depths, stencils) = data.split_at(pixels * 4);
                for y in first..last {
                    for x in 0..width {
                        let i = ((height - 1 - y) * width + x) as usize;
                        let d24 = u32::from_le_bytes(depths[i * 4..i * 4 + 4].try_into().unwrap()) & 0xFF_FFFF;
                        let out = morton_offset(x, y, width, sample) as usize;
                        match sample {
                            2 => {
                                let d16 = ((d24 as u64 * 0xFFFF + 0x7F_FFFF) / 0xFF_FFFF) as u16;
                                bytes[out..out + 2].copy_from_slice(&d16.to_le_bytes());
                            }
                            3 => bytes[out..out + 3].copy_from_slice(&d24.to_le_bytes()[..3]),
                            _ => {
                                let word = d24 | (stencils[i] as u32) << 24;
                                bytes[out..out + 4].copy_from_slice(&word.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        let row = s.row_bytes() as usize;
        let addr = s.addr;
        let rows = first as usize * row..last as usize * row;
        let drawn = bytes[rows.clone()].to_vec();
        // what was written over the buffer since the image last matched it
        // came after the drawing and stays, the console's GPU had put its
        // pixels in memory first. the CPU's writes and the files services
        // read have them written back first, see guard_writes, this keeps
        // what anything else wrote, but for bytes it left as they were
        if s.shadow.len() == bytes.len() {
            let mut now = vec![0u8; rows.len()];
            memory.read(addr + rows.start as u32, &mut now);
            for ((out, &held), &was) in bytes[rows.clone()].iter_mut().zip(&now).zip(&s.shadow[rows.clone()]) {
                if held != was {
                    *out = held;
                }
            }
        }
        memory.write(addr + rows.start as u32, &bytes[rows.clone()]);
        let surface = &mut self.surfaces[index];
        if surface.shadow.len() == bytes.len() {
            // the image still matches what memory held where it did before,
            // and the rows written hold what it drew, what memory kept
            // instead shows as a change the next time it is looked up
            surface.shadow[rows].copy_from_slice(&drawn);
        } else {
            // what memory holds now, the rows drawn and the others as they were
            memory.read(addr, &mut bytes);
            surface.shadow = bytes;
        }
        surface.dirty = None;
        surface.guarded = None;
        surface.write_guarded = None;
        surface.cleared = None;
    }
}

/// the range of memory to guard for the rows drawn, which joins the rows
/// guarded before, none when they all were.
fn widen(guarded: &mut Option<(u32, u32)>, dirty: Option<(u32, u32)>, addr: u32, row: u32) -> Option<(u32, u32)> {
    let (start, end) = dirty?;
    if guarded.is_some_and(|(from, to)| from <= start && end <= to) {
        return None;
    }
    // all of it, which overlaps what was guarded before and joins it
    let (start, end) = guarded.map_or((start, end), |(from, to)| (from.min(start), to.max(end)));
    *guarded = Some((start, end));
    Some((addr + start * row, (end - start) * row))
}

/// a color format as the transfer shader numbers it, the register's value.
pub(crate) fn format_index(format: ColorFormat) -> i32 {
    match format {
        ColorFormat::Rgba8 => 0,
        ColorFormat::Rgb8 => 1,
        ColorFormat::Rgb565 => 2,
        ColorFormat::Rgb5A1 => 3,
        ColorFormat::Rgba4 => 4,
    }
}

impl Drop for Hardware {
    fn drop(&mut self) {
        // the compiler first, it uses the cache, the layout and the modules
        let unclaimed = self.compiler.finish();
        // SAFETY: waits for the GPU before anything it uses goes
        unsafe {
            let _ = self.device.device_wait_idle();
            save_pipeline_cache(&self.device, self.pipeline_cache);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            for texture in self.textures.values() {
                self.destroy_image(&texture.image);
            }
            for copy in self.copies.values() {
                self.destroy_image(&copy.image);
            }
            for surface in &self.surfaces {
                self.destroy_image(&surface.image);
                if let Some(native) = &surface.native {
                    self.destroy_image(native);
                }
                if let Some(upright) = &surface.upright {
                    self.destroy_image(&upright.image);
                }
            }
            self.destroy_image(&self.blank);
            let captures = self.surfaces.iter().flat_map(|s| [&s.capture, &s.screen]).filter_map(|c| c.as_ref().map(|c| &c.buffer));
            let rings = self.free.iter().chain(&self.in_flight).map(|frame| &frame.ring);
            let buffers = [Some(&self.ring), self.readback.as_ref()].into_iter().flatten().chain(rings);
            for buffer in buffers.chain(captures) {
                self.device.destroy_buffer(buffer.buffer, None);
                self.device.free_memory(buffer.memory, None);
            }
            for compute in [&self.transfer, &self.depth, &self.upright, &self.upright_image].into_iter().flatten() {
                self.device.destroy_pipeline(compute.pipeline, None);
                self.device.destroy_pipeline_layout(compute.layout, None);
                self.device.destroy_descriptor_set_layout(compute.set_layout, None);
                self.device.destroy_shader_module(compute.shader, None);
            }
            if let Some(samples) = &self.samples {
                self.device.destroy_buffer(samples.buffer, None);
                self.device.free_memory(samples.memory, None);
            }
            for &pipeline in self.pipelines.values().chain(&unclaimed) {
                self.device.destroy_pipeline(pipeline, None);
            }
            for &sampler in self.samplers.values() {
                self.device.destroy_sampler(sampler, None);
            }
            self.device.destroy_shader_module(self.vertex_shader, None);
            self.device.destroy_shader_module(self.fragment_shader, None);
            self.device.destroy_shader_module(self.depth_fragment_shader, None);
            self.device.destroy_shader_module(self.shade_shader, None);
            for &module in self.modules.values() {
                self.device.destroy_shader_module(module, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_fence(self.fence, None);
            for frame in self.free.iter().chain(&self.in_flight) {
                self.device.destroy_fence(frame.fence, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            // the device goes with the last of the presenter and this
        }
    }
}

// SAFETY: the mapped pointers belong to buffers only this value uses, and
// Vulkan handles may move between threads
unsafe impl Send for Hardware {}

/// the part of a buffer's upright picture a screen shows. a row of the
/// buffer is a column of the picture, and a row's first pixels are the
/// bottom of the screen, a longer row reaches past its top.
fn crop(image: Arc<Vec<u8>>, screen: ScreenRef, scale: u32) -> Arc<Vec<u8>> {
    let (row_pixels, rows) = screen.size;
    let (first, count) = screen.rows;
    if first == 0 && count == rows && screen.columns == row_pixels {
        return image;
    }
    let width = (rows * scale) as usize;
    let (left, right) = ((first * scale) as usize, ((first + count) * scale) as usize);
    let (top, bottom) = (((row_pixels - screen.columns) * scale) as usize, (row_pixels * scale) as usize);
    let mut out = Vec::with_capacity((right - left) * (bottom - top) * 4);
    for y in top..bottom {
        out.extend_from_slice(&image[(y * width + left) * 4..(y * width + right) * 4]);
    }
    Arc::new(out)
}

/// the Vulkan logic op for one of the PICA's.
fn logic_op(op: LogicOp) -> vk::LogicOp {
    match op {
        LogicOp::Clear => vk::LogicOp::CLEAR,
        LogicOp::And => vk::LogicOp::AND,
        LogicOp::AndReverse => vk::LogicOp::AND_REVERSE,
        LogicOp::Copy => vk::LogicOp::COPY,
        LogicOp::Set => vk::LogicOp::SET,
        LogicOp::CopyInverted => vk::LogicOp::COPY_INVERTED,
        LogicOp::Noop => vk::LogicOp::NO_OP,
        LogicOp::Invert => vk::LogicOp::INVERT,
        LogicOp::Nand => vk::LogicOp::NAND,
        LogicOp::Or => vk::LogicOp::OR,
        LogicOp::Nor => vk::LogicOp::NOR,
        LogicOp::Xor => vk::LogicOp::XOR,
        LogicOp::Equivalent => vk::LogicOp::EQUIVALENT,
        LogicOp::AndInverted => vk::LogicOp::AND_INVERTED,
        LogicOp::OrReverse => vk::LogicOp::OR_REVERSE,
        LogicOp::OrInverted => vk::LogicOp::OR_INVERTED,
    }
}

/// whether the renderer can draw on a GPU, Vulkan 1.3, push descriptors,
/// depth clamping and a D24S8 depth format.
fn renders_on(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
    // SAFETY: queries on a GPU of the instance
    let (properties, features, depth, extensions) = unsafe {
        (
            instance.get_physical_device_properties(device),
            instance.get_physical_device_features(device),
            instance.get_physical_device_format_properties(device, DEPTH_FORMAT),
            instance.enumerate_device_extension_properties(device).unwrap_or_default(),
        )
    };
    let push = extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::push_descriptor::NAME));
    properties.api_version >= vk::API_VERSION_1_3
        && features.depth_clamp == vk::TRUE
        && depth.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        && push
}

/// whether the renderer can draw on a GPU through a family of its queues,
/// which has to do compute as well, for display transfers.
pub(crate) fn can_render(instance: &ash::Instance, device: vk::PhysicalDevice, family: u32) -> bool {
    // SAFETY: as above
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let both = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    families.get(family as usize).is_some_and(|f| f.queue_flags.contains(both)) && renders_on(instance, device)
}

/// a device the renderer draws with, on a GPU can_render or pick chose,
/// with the extensions given besides its own. it owns the instance from then
/// on, destroying it when it goes, which stays the caller's when it fails.
pub(crate) fn render_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
    extra: &[*const std::ffi::c_char],
) -> Result<SharedDevice, String> {
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
    let mut extensions = vec![ash::khr::push_descriptor::NAME.as_ptr()];
    extensions.extend_from_slice(extra);
    // SAFETY: the physical device came from this instance
    let supported = unsafe { instance.get_physical_device_features(physical) };
    let logic_ops = supported.logic_op == vk::TRUE;
    let dynamic = dynamic_state(instance, physical, logic_ops);
    if dynamic.blend {
        extensions.push(ash::ext::extended_dynamic_state3::NAME.as_ptr());
    }
    if dynamic.logic_op {
        extensions.push(ash::ext::extended_dynamic_state2::NAME.as_ptr());
    }
    let mut blend_state = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
        .extended_dynamic_state3_color_blend_enable(true)
        .extended_dynamic_state3_color_blend_equation(true)
        .extended_dynamic_state3_color_write_mask(true)
        .extended_dynamic_state3_logic_op_enable(dynamic.logic_op);
    let mut logic_op_state = vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default().extended_dynamic_state2_logic_op(true);
    // what the GPU shades counted along with its times
    let statistics = std::env::var_os("ZAKURO_GPU_TIMES").is_some() && supported.pipeline_statistics_query == vk::TRUE;
    let features = vk::PhysicalDeviceFeatures::default()
        .depth_clamp(true)
        .logic_op(logic_ops)
        .pipeline_statistics_query(statistics);
    // the cache control, to make a pipeline only when the cache on disk
    // has it, every Vulkan 1.3 device does
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true)
        .pipeline_creation_cache_control(true);
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut features13);
    if dynamic.blend {
        info = info.push_next(&mut blend_state);
    }
    if dynamic.logic_op {
        info = info.push_next(&mut logic_op_state);
    }
    // SAFETY: the caller checked the device has all of this
    let device = unsafe { instance.create_device(physical, &info, None) }.map_err(vk_error("create a device"))?;
    // SAFETY: a queue the device was made with
    let queue = unsafe { device.get_device_queue(family, 0) };
    let (_entry, instance) = (entry.clone(), instance.clone());
    Ok(SharedDevice { _entry, instance, physical, family, device, queue, logic_ops, statistics, dynamic })
}

/// which of the blend state the device lets draws set as they go, none
/// with ZAKURO_STATIC_BLEND set, to compare the two.
fn dynamic_state(instance: &ash::Instance, physical: vk::PhysicalDevice, logic_ops: bool) -> Dynamic {
    if std::env::var_os("ZAKURO_STATIC_BLEND").is_some() {
        return Dynamic::default();
    }
    // SAFETY: the physical device came from this instance
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has = |name: &std::ffi::CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
    let (three, two) = (has(ash::ext::extended_dynamic_state3::NAME), has(ash::ext::extended_dynamic_state2::NAME));
    let mut supported3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
    let mut supported2 = vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if three {
            features = features.push_next(&mut supported3);
        }
        if two {
            features = features.push_next(&mut supported2);
        }
        // SAFETY: as above, with a chain of structures that live as long
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
    }
    let blend = three
        && supported3.extended_dynamic_state3_color_blend_enable == vk::TRUE
        && supported3.extended_dynamic_state3_color_blend_equation == vk::TRUE
        && supported3.extended_dynamic_state3_color_write_mask == vk::TRUE;
    let logic_op = blend
        && logic_ops
        && two
        && supported2.extended_dynamic_state2_logic_op == vk::TRUE
        && supported3.extended_dynamic_state3_logic_op_enable == vk::TRUE;
    Dynamic { blend, logic_op }
}

/// a device of the renderer's own, on the GPU pick chooses.
pub(crate) fn own_device() -> Result<SharedDevice, String> {
    // SAFETY: loads the system's Vulkan library, which has no other
    // requirements
    let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader, {e}"))?;
    let app = vk::ApplicationInfo::default().application_name(c"zakuro").api_version(vk::API_VERSION_1_3);
    // SAFETY: a plain instance with no layers or extensions
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
        .map_err(vk_error("create an instance"))?;
    let made = pick(&instance).and_then(|(physical, family)| render_device(&entry, &instance, physical, family, &[]));
    if made.is_err() {
        // SAFETY: nothing made from the instance is left
        unsafe { instance.destroy_instance(None) };
    }
    made
}

/// the device to draw with and its graphics queue family, a discrete GPU
/// when there is one.
fn pick(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), String> {
    // SAFETY: queries on an instance we own
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(vk_error("list the GPUs"))?;
    let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
    for device in devices {
        // SAFETY: as above
        let (properties, families) =
            unsafe { (instance.get_physical_device_properties(device), instance.get_physical_device_queue_family_properties(device)) };
        let Some(family) = families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS)) else { continue };
        if !renders_on(instance, device) {
            continue;
        }
        let rank = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        };
        if best.is_none_or(|(best, ..)| rank < best) {
            best = Some((rank, device, family as u32));
        }
    }
    best.map(|(_, device, family)| (device, family))
        .ok_or_else(|| "no GPU with Vulkan 1.3, push descriptors and a D24S8 depth format".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a block gathered with the words of the last one staged is read from
    /// where that went, any other is staged anew, and nothing is read from
    /// a batch handed over.
    #[test]
    fn a_block_is_staged_again_only_when_its_words_change() {
        let mut block = Block::default();
        let gather = |block: &mut Block, words: &[u32]| {
            block.words.clear();
            block.words.extend_from_slice(words);
        };
        gather(&mut block, &[1, 2, 3]);
        assert_eq!(block.staged(), None, "nothing staged yet");
        block.staged_at(64);
        gather(&mut block, &[1, 2, 3]);
        assert_eq!(block.staged(), Some(64));
        gather(&mut block, &[1, 2, 4]);
        assert_eq!(block.staged(), None, "a word changed");
        block.staged_at(128);
        gather(&mut block, &[1, 2, 4]);
        assert_eq!(block.staged(), Some(128));
        gather(&mut block, &[1, 2, 3]);
        assert_eq!(block.staged(), None, "only the last block is kept");
        gather(&mut block, &[1, 2, 4]);
        block.at = None;
        assert_eq!(block.staged(), None, "the batch was handed over");
    }

    /// what a surface keeps of guest memory, through a pixel standing in for
    /// its shadow or not, is what the plain bytes would be, after uploads of
    /// memory of one pixel or not, fills over all of it or rows of it, and
    /// write-backs that have the bytes written out.
    #[test]
    fn a_pixel_standing_in_for_the_shadow_keeps_what_the_bytes_would() {
        let mut seed = 3u32;
        let mut random = move |range: usize| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as usize % range
        };
        let kinds = [Kind::Color(ColorFormat::Rgba8), Kind::Color(ColorFormat::Rgb8), Kind::Color(ColorFormat::Rgb565), Kind::Depth(2), Kind::Depth(3), Kind::Depth(4)];
        for kind in kinds {
            let (width, height) = (16, 24);
            let nothing = Image { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() };
            let mut surface = Surface {
                image: nothing,
                addr: 0,
                width,
                height,
                kind,
                tiled: true,
                shadow: Vec::new(),
                pixel: None,
                checked: false,
                dirty: None,
                guarded: None,
                write_guarded: None,
                cleared: None,
                capture: None,
                native: None,
                screen: None,
                upright: None,
                generation: 0,
            };
            let (size, bpp, row) = (surface.size() as usize, kind.bytes() as usize, surface.row_bytes() as usize);
            // the bytes the shadow stands for, none before the first upload
            let mut bytes: Option<Vec<u8>> = None;
            let mut pixel: Vec<u8> = vec![0; bpp];
            for step in 0..300 {
                // a pixel of its own, or the last one again
                if random(2) == 0 {
                    pixel = (0..bpp).map(|_| random(256) as u8).collect();
                }
                match random(5) {
                    0 | 1 => {
                        // an upload of memory, of one pixel or not
                        let mut uploaded = pixel.repeat(size / bpp);
                        if random(2) == 0 {
                            uploaded[random(size)] ^= 1 + random(255) as u8;
                        }
                        surface.keep(&uploaded);
                        bytes = Some(uploaded);
                    }
                    2 => {
                        // a fill over all of it
                        surface.keep_pixel(&pixel);
                        bytes = Some(pixel.repeat(size / bpp));
                    }
                    3 => {
                        // a fill over rows of tiles
                        let from = random(height as usize / 8) * 8;
                        let to = from + 8 * (1 + random((height as usize - from) / 8));
                        surface.keep_rows(from * row..to * row, &pixel);
                        if let Some(bytes) = &mut bytes {
                            bytes[from * row..to * row].copy_from_slice(&pixel.repeat((to - from) * row / bpp));
                        }
                    }
                    _ => {
                        // a write-back, which reads the bytes and changes rows
                        assert_eq!(*surface.shadow(), bytes.clone().unwrap_or_default(), "{kind:?} at step {step}");
                        if let Some(bytes) = &mut bytes {
                            let at = random(height as usize) * row;
                            let drawn: Vec<u8> = (0..row).map(|_| random(256) as u8).collect();
                            surface.shadow()[at..at + row].copy_from_slice(&drawn);
                            bytes[at..at + row].copy_from_slice(&drawn);
                        }
                    }
                }
                match &bytes {
                    Some(bytes) => {
                        assert!(surface.shadows(0..size, bytes), "{kind:?} at step {step}");
                        let mut changed = bytes.clone();
                        changed[random(size)] ^= 0x10;
                        assert!(!surface.shadows(0..size, &changed), "{kind:?} at step {step}");
                        // rows of it, as a screen shows them
                        let (first, rows) = (random(height as usize), random(4) + 1);
                        let rows = first * row..((first + rows) * row).min(size);
                        assert!(surface.shadows(rows.clone(), &bytes[rows.clone()]), "{kind:?} at step {step}");
                        let past = [&bytes[rows.start..], &bytes[..row]].concat();
                        assert!(!surface.shadows(rows.start..size + row, &past), "{kind:?} past the end");
                    }
                    None => assert!(!surface.shadows(0..size, &vec![0; size]), "{kind:?} before an upload"),
                }
            }
        }
    }

    /// the generic pipelines made at the start are every one a draw can ask
    /// for while the blend state is dynamic.
    #[test]
    fn the_generic_pipelines_made_at_the_start_are_all_a_draw_needs() {
        let made: QuickSet<PipelineKey> = generic_keys(true).into_iter().collect();
        let mut seed = 5u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        for _ in 0..1000 {
            let mut fragment: [u32; FRAGMENT_CONSTANTS] = std::array::from_fn(|_| random());
            // whether the draw is lit
            fragment[26] &= 1;
            for vertex in [VertexStage::Placed, VertexStage::Interpreted] {
                let key = PipelineKey { blend: None, logic_op: None, mask: 0xF, depth: random() & 1 != 0, vertex, fragment };
                // as pipeline_for asks for it
                assert!(made.contains(&PipelineKey { fragment: generic(&key.fragment), ..key }));
            }
        }
    }
}
