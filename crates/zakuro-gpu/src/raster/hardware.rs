//! filling triangles on the host's GPU through Vulkan. the CPU still works
//! out the geometry the way the software path does, shading, clipping and
//! culling, and the GPU does what costs, the pixels, texturing, lighting,
//! the combiners, the tests and blending.
//!
//! guest memory stays where images live between command lists. a buffer a
//! list draws into goes up to the GPU the first time the list needs it,
//! unless it is still as the GPU left it, and comes back down when the list
//! ends or when something is about to read it as a texture.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;

use super::{BoundTexture, DepthMap, Screen, Wrap, TEXTURE_UNIT_BASES};
use crate::format::{morton_offset, ColorFormat};
use crate::lighting::{Lighting, Tables};
use crate::registers::*;
use crate::GpuMemory;

const VERTEX_SPIRV: &[u8] = include_bytes!("../../shaders/raster.vert.spv");
const FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/raster.frag.spv");

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D24_UNORM_S8_UINT;

/// where a batch keeps its vertices, uniforms and uploads.
const RING_SIZE: u64 = 128 << 20;
/// a batch past this much is run before the next draw, so a draw always
/// finds room.
const RING_FLUSH: u64 = RING_SIZE / 2;

/// a vertex is six vectors, position, color, texture coordinates 0 and 1,
/// coordinate 2 with the mapped depth, the quaternion and the view vector.
const VERTEX_SIZE: usize = 24 * 4;
/// the words of a draw's uniform block, the combiners, the texture units
/// and the flags, then the lighting.
const UNIFORM_WORDS: usize = 12 * 4 + 4 + 3 * 4 + 4 + 328;
const UNIFORM_SIZE: u64 = (UNIFORM_WORDS * 4) as u64;
/// 24 lighting tables of 256 entries, each a value and a step.
const TABLES_SIZE: u64 = 24 * 256 * 8;

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
    pub(super) triangles: &'a [[Screen; 3]],
    pub(super) textures: &'a [Option<BoundTexture>; 3],
    pub(super) lighting: Option<&'a Lighting>,
    pub(super) tables: &'a Tables,
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
    /// the guest's bytes the last time the image matched them.
    shadow: Vec<u8>,
    /// the image is known to match guest memory in this batch.
    checked: bool,
    /// drawn into since guest memory last got the image.
    dirty: bool,
}

impl Surface {
    fn size(&self) -> u32 {
        self.width * self.height * self.kind.bytes()
    }

    fn overlaps(&self, addr: u32, len: u32) -> bool {
        addr < self.addr + self.size() && self.addr < addr + len
    }
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
    /// red, green, blue and alpha writes, one bit each.
    mask: u32,
    depth: bool,
}

pub struct Hardware {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    push: ash::khr::push_descriptor::Device,
    queue: vk::Queue,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    uniform_alignment: u64,
    storage_alignment: u64,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    samplers: HashMap<(bool, Wrap, Wrap), vk::Sampler>,
    ring: Buffer,
    used: u64,
    readback: Option<Buffer>,
    surfaces: Vec<Surface>,
    /// by the address of the decoded texels.
    textures: HashMap<usize, Texture>,
    /// what unused texture units sample.
    blank: Image,
    /// the tables' generation and where the batch copied them.
    tables: Option<(u64, u64)>,
    recording: bool,
    /// uploads waiting for a barrier before anything reads them.
    uploads: bool,
    /// the color and depth surfaces being rendered into.
    rendering: Option<(usize, Option<usize>)>,
    batch: u64,
    name: String,
}

impl Hardware {
    pub fn new() -> Result<Hardware, String> {
        // SAFETY: loads the system's Vulkan library, which has no other
        // requirements
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader, {e}"))?;
        let app = vk::ApplicationInfo::default().application_name(c"zakuro").api_version(vk::API_VERSION_1_3);
        // SAFETY: a plain instance with no layers or extensions
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }
            .map_err(vk_error("create an instance"))?;
        match Hardware::with_instance(entry, instance.clone()) {
            Ok(hardware) => Ok(hardware),
            Err(error) => {
                // SAFETY: nothing was made from the instance that outlives it
                unsafe { instance.destroy_instance(None) };
                Err(error)
            }
        }
    }

    fn with_instance(entry: ash::Entry, instance: ash::Instance) -> Result<Hardware, String> {
        let (physical, family) = pick(&instance)?;
        // SAFETY: the physical device came from this instance
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let extensions = [ash::khr::push_descriptor::NAME.as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default().depth_clamp(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut features13);
        // SAFETY: pick checked the device has all of this
        let device = unsafe { instance.create_device(physical, &info, None) }.map_err(vk_error("create a device"))?;
        let push = ash::khr::push_descriptor::Device::new(&instance, &device);

        // SAFETY: everything below is made from the device just created,
        // with create infos that live as long as each call
        unsafe {
            let queue = device.get_device_queue(family, 0);
            let memory_types = instance.get_physical_device_memory_properties(physical);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(family),
                    None,
                )
                .map_err(vk_error("create a command pool"))?;
            let commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vk_error("allocate a command buffer"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vk_error("create a fence"))?;

            let bindings: Vec<_> = (0..5)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(match binding {
                            0..=2 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            3 => vk::DescriptorType::UNIFORM_BUFFER,
                            _ => vk::DescriptorType::STORAGE_BUFFER,
                        })
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
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

            let mut hardware = Hardware {
                ring: Buffer {
                    buffer: vk::Buffer::null(),
                    memory: vk::DeviceMemory::null(),
                    size: 0,
                    mapped: std::ptr::null_mut(),
                    incoherent: false,
                },
                blank: Image { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() },
                _entry: entry,
                instance,
                device,
                push,
                queue,
                memory_types,
                uniform_alignment: properties.limits.min_uniform_buffer_offset_alignment.max(16),
                storage_alignment: properties.limits.min_storage_buffer_offset_alignment.max(16),
                pool,
                commands,
                fence,
                set_layout,
                layout,
                vertex_shader,
                fragment_shader,
                pipelines: HashMap::new(),
                samplers: HashMap::new(),
                used: 0,
                readback: None,
                surfaces: Vec::new(),
                textures: HashMap::new(),
                tables: None,
                recording: false,
                uploads: false,
                rendering: None,
                batch: 0,
                name,
            };
            hardware.ring = hardware.buffer(
                RING_SIZE,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                false,
            )?;
            hardware.begin()?;
            hardware.blank = hardware.image(
                1,
                1,
                COLOR_FORMAT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
            )?;
            Ok(hardware)
        }
    }

    /// the GPU it draws with.
    pub fn name(&self) -> &str {
        &self.name
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
        }
        Ok(())
    }

    /// makes everything recorded so far visible to everything after it.
    fn barrier(&self) {
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        // SAFETY: recording is on whenever this is called
        unsafe { self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&barrier)) };
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

    fn ring(&mut self, offset: u64, size: u64) -> &mut [u8] {
        // SAFETY: the ring is mapped for its whole size and stage kept the
        // range inside it, the GPU is not reading it while this batch is
        // being recorded
        unsafe { std::slice::from_raw_parts_mut(self.ring.mapped.add(offset as usize), size as usize) }
    }

    /// the surface for a guest buffer, matching guest memory.
    fn surface<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, width: u32, height: u32, kind: Kind) -> Result<usize, String> {
        let index = match self
            .surfaces
            .iter()
            .position(|s| s.addr == addr && s.width == width && s.height == height && s.kind == kind)
        {
            Some(index) => index,
            None => {
                let (format, usage, aspect) = match kind {
                    Kind::Color(_) => (COLOR_FORMAT, vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR),
                    Kind::Depth(_) => (
                        DEPTH_FORMAT,
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    ),
                };
                let usage = usage | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
                let image = self.image(width, height, format, usage, aspect)?;
                self.uploads = true;
                self.surfaces.push(Surface { image, addr, width, height, kind, shadow: Vec::new(), checked: false, dirty: false });
                self.surfaces.len() - 1
            }
        };
        if self.surfaces[index].checked {
            return Ok(index);
        }
        // what another surface over the same memory drew has to reach guest
        // memory before this one reads it
        let size = self.surfaces[index].size();
        let others: Vec<usize> = (0..self.surfaces.len())
            .filter(|&i| i != index && self.surfaces[i].dirty && self.surfaces[i].overlaps(addr, size))
            .collect();
        if !others.is_empty() {
            self.run(memory, others)?;
            self.begin()?;
        }
        let mut bytes = vec![0; size as usize];
        memory.read(addr, &mut bytes);
        if bytes != self.surfaces[index].shadow {
            self.upload(index, &bytes)?;
            self.surfaces[index].shadow = bytes;
        }
        self.surfaces[index].checked = true;
        Ok(index)
    }

    /// copies a guest buffer's bytes into its surface.
    fn upload(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        self.end_rendering();
        let (width, height, kind) = {
            let s = &self.surfaces[index];
            (s.width, s.height, s.kind)
        };
        let pixels = (width * height) as u64;
        let image = self.surfaces[index].image.image;
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
                        let at = morton_offset(x, y, width, bpp as u32) as usize;
                        let rgba = format.decode(&bytes[at..at + bpp]);
                        let out = (((height - 1 - y) * width + x) * 4) as usize;
                        staging[out..out + 4].copy_from_slice(&rgba);
                    }
                }
                let regions = [region(offset, vk::ImageAspectFlags::COLOR)];
                // SAFETY: recording, and the regions lie inside the ring
                unsafe {
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
                    self.device.cmd_copy_buffer_to_image(self.commands, self.ring.buffer, image, vk::ImageLayout::GENERAL, &regions)
                };
            }
        }
        self.uploads = true;
        Ok(())
    }

    /// the texture's image, uploaded the first time it is drawn with.
    fn texture(&mut self, bound: &BoundTexture) -> Result<vk::ImageView, String> {
        let key = Arc::as_ptr(&bound.texels) as *const u8 as usize;
        if let Some(texture) = self.textures.get_mut(&key) {
            texture.used = self.batch;
            return Ok(texture.image.view);
        }
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
    fn tables(&mut self, tables: &Tables) -> Result<u64, String> {
        if let Some((generation, offset)) = self.tables {
            if generation == tables.generation() {
                return Ok(offset);
            }
        }
        let offset = self.stage(TABLES_SIZE, self.storage_alignment)?;
        let staging = self.ring(offset, TABLES_SIZE);
        let values = tables.entries().iter().flatten().flatten();
        for (out, value) in staging.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *out = value.to_le_bytes();
        }
        self.tables = Some((tables.generation(), offset));
        Ok(offset)
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<vk::Pipeline, String> {
        if let Some(&pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline);
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex_shader)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_shader)
                .name(c"main"),
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
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        // the CPU culled already, and clipped what matters, depth comes
        // from the shader
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(true)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
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
        let mut attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::from_raw(key.mask));
        if let Some(config) = key.blend {
            attachment = attachment
                .blend_enable(true)
                .color_blend_op(equation(config))
                .alpha_blend_op(equation(config >> 8))
                .src_color_blend_factor(factor(config >> 16))
                .dst_color_blend_factor(factor(config >> 20))
                .src_alpha_blend_factor(factor(config >> 24))
                .dst_alpha_blend_factor(factor(config >> 28));
        }
        let attachments = [attachment];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [
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
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [COLOR_FORMAT];
        let depth_format = if key.depth { DEPTH_FORMAT } else { vk::Format::UNDEFINED };
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(depth_format);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .push_next(&mut rendering);
        // SAFETY: every state the create info points at lives until it
        // returns
        let pipeline = unsafe { self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, error)| format!("could not create a pipeline, {error}"))?[0];
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// a memory fill wrote bytes at addr. a surface it covered with one
    /// value is cleared to it on the GPU as well, rather than uploaded again
    /// the next time it is drawn into.
    pub(crate) fn filled(&mut self, addr: u32, bytes: &[u8]) -> Result<(), String> {
        let end = addr as u64 + bytes.len() as u64;
        for index in 0..self.surfaces.len() {
            let surface = &self.surfaces[index];
            let size = surface.size() as usize;
            if surface.addr < addr || surface.addr as u64 + size as u64 > end {
                continue;
            }
            let offset = (surface.addr - addr) as usize;
            let filled = &bytes[offset..offset + size];
            let bpp = surface.kind.bytes() as usize;
            // a pattern that does not line up with the pixels leaves them
            // different from each other, and uploading handles that
            if !filled.iter().enumerate().take(12).all(|(i, &b)| b == filled[i % bpp]) {
                continue;
            }
            let pixel = filled[..bpp].to_vec();
            let filled = filled.to_vec();
            let (image, kind) = (surface.image.image, surface.kind);
            self.begin()?;
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
                        self.device.cmd_clear_depth_stencil_image(self.commands, image, vk::ImageLayout::GENERAL, &value, &range);
                    }
                }
            }
            self.uploads = true;
            let surface = &mut self.surfaces[index];
            surface.shadow = filled;
            surface.dirty = false;
            surface.checked = false;
        }
        Ok(())
    }

    /// makes sure guest memory holds what the GPU drew over a range, before
    /// something reads it.
    pub(crate) fn prepare_read<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        if self.surfaces.iter().any(|s| s.dirty && s.overlaps(addr, len)) {
            self.sync(memory, addr, len)?;
        }
        Ok(())
    }

    /// a fill is about to write a range, whatever the GPU drew over the part
    /// of a buffer it leaves alone has to come down first.
    pub(crate) fn before_fill<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        let end = addr as u64 + len as u64;
        let partial: Vec<usize> = (0..self.surfaces.len())
            .filter(|&i| {
                let s = &self.surfaces[i];
                s.dirty && s.overlaps(addr, len) && (s.addr < addr || s.addr as u64 + s.size() as u64 > end)
            })
            .collect();
        if !partial.is_empty() {
            self.run(memory, partial)?;
        }
        Ok(())
    }

    /// records one draw.
    pub(super) fn draw<M: GpuMemory>(&mut self, memory: &mut M, draw: &Draw) -> Result<(), String> {
        let [left, bottom, right, top] = draw.scissor;
        let (left, bottom) = (left.max(0), bottom.max(0));
        let (right, top) = (right.min(draw.width as i32), top.min(draw.height as i32));
        if draw.triangles.is_empty() || right <= left || top <= bottom {
            return Ok(());
        }
        if self.used > RING_FLUSH {
            self.run(memory, Vec::new())?;
        }
        self.begin()?;

        // a flush looking one of them up drops what the other had checked
        let (color, depth) = loop {
            let color = self.surface(memory, draw.target, draw.width, draw.height, Kind::Color(draw.format))?;
            let depth = match draw.depth {
                Some((addr, bytes)) => Some(self.surface(memory, addr, draw.width, draw.height, Kind::Depth(bytes))?),
                None => None,
            };
            if self.surfaces[color].checked && depth.is_none_or(|d| self.surfaces[d].checked) {
                break (color, depth);
            }
        };

        let mut views = [self.blank.view; 3];
        let mut samplers = [vk::Sampler::null(); 3];
        let mut enabled = 0;
        for (unit, bound) in draw.textures.iter().enumerate() {
            match bound {
                Some(bound) => {
                    views[unit] = self.texture(bound)?;
                    samplers[unit] = self.sampler(bound.linear, bound.wrap_s, bound.wrap_t)?;
                    enabled |= 1 << unit;
                }
                None => samplers[unit] = self.sampler(false, Wrap::ClampToEdge, Wrap::ClampToEdge)?,
            }
        }
        let tables = match draw.lighting {
            Some(_) => self.tables(draw.tables)?,
            None => 0,
        };

        // the vertices, with the target's pixels as clip space and w kept
        // for perspective
        let vertex_count = draw.triangles.len() * 3;
        let vertex_bytes = (vertex_count * VERTEX_SIZE) as u64;
        let vertex_offset = self.stage(vertex_bytes, 16)?;
        let (width, height) = (draw.width as f32, draw.height as f32);
        let depth_map = draw.depth_map;
        // a pixel whose center sits exactly on an edge goes to the triangle
        // on its right, or above it for a flat edge, on the PICA, and in
        // Vulkan to the one on its right or further down the image, so the
        // images run bottom up
        let staging = self.ring(vertex_offset, vertex_bytes);
        for (out, v) in staging.as_chunks_mut::<VERTEX_SIZE>().0.iter_mut().zip(draw.triangles.iter().flatten()) {
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

        let r = draw.registers;
        let mut words = Vec::with_capacity(UNIFORM_WORDS);
        for base in STAGE_REGISTERS {
            words.extend([r[base], r[base + 1], r[base + 2], r[base + 3], r[base + 4], 0, 0, 0]);
        }
        let texture_config = (r[REG_TEXTURE_CONFIG] & !7) | enabled;
        words.extend([r[REG_UPDATE_BUFFER], r[REG_BUFFER_COLOR], r[REG_ALPHA_TEST], texture_config]);
        for base in TEXTURE_UNIT_BASES {
            words.extend([r[base + 2], r[base], 0, 0]);
        }
        words.extend([depth_map.w_buffer as u32, draw.lighting.is_some() as u32, 0, 0]);
        match draw.lighting {
            Some(lighting) => lighting.pack(&mut words),
            None => words.resize(UNIFORM_WORDS, 0),
        }
        debug_assert_eq!(words.len(), UNIFORM_WORDS);
        let uniform_offset = self.stage(UNIFORM_SIZE, self.uniform_alignment)?;
        let staging = self.ring(uniform_offset, UNIFORM_SIZE);
        for (out, word) in staging.as_chunks_mut::<4>().0.iter_mut().zip(&words) {
            *out = word.to_le_bytes();
        }

        // what the draw may change
        let color_mask = if r[REG_COLOR_BUFFER_WRITE] != 0 { (r[REG_DEPTH_COLOR_MASK] >> 8) & 0xF } else { 0 };
        let mask = r[REG_DEPTH_COLOR_MASK];
        let writable = r[REG_DEPTH_STENCIL_WRITE] != 0;
        let depth_test = mask & 1 != 0;
        let depth_write = writable && mask & (1 << 12) != 0;
        let stencil_test = draw.depth.is_some_and(|(_, bytes)| bytes == 4) && r[REG_STENCIL_TEST] & 1 != 0;
        let blend = (r[REG_COLOR_OPERATION] & 0x100 != 0).then_some(r[REG_BLEND_FUNC]);
        let pipeline = self.pipeline(PipelineKey { blend, mask: color_mask, depth: depth.is_some() })?;

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
            self.surfaces[color].dirty = true;
        }
        if let Some(depth) = depth {
            if depth_write || (stencil_test && writable) {
                self.surfaces[depth].dirty = true;
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
        let writes = [
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

        let commands = self.commands;
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
            device.cmd_set_viewport(
                commands,
                0,
                &[vk::Viewport { x: 0.0, y: 0.0, width, height, min_depth: 0.0, max_depth: 1.0 }],
            );
            device.cmd_set_scissor(
                commands,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: left, y: bottom },
                    extent: vk::Extent2D { width: (right - left) as u32, height: (top - bottom) as u32 },
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
            self.push.cmd_push_descriptor_set(commands, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &writes);
            device.cmd_bind_vertex_buffers(commands, 0, &[self.ring.buffer], &[vertex_offset]);
            device.cmd_draw(commands, vertex_count as u32, 1, 0, 0);
        }
        Ok(())
    }

    fn begin_rendering(&mut self, color: usize, depth: Option<usize>) {
        let surface = &self.surfaces[color];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: surface.width, height: surface.height },
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
        unsafe { self.device.cmd_begin_rendering(self.commands, &info) };
        self.rendering = Some((color, depth));
    }

    /// runs what the batch recorded and writes everything it drew back to
    /// guest memory.
    pub(crate) fn flush<M: GpuMemory>(&mut self, memory: &mut M) -> Result<(), String> {
        let dirty = (0..self.surfaces.len()).filter(|&i| self.surfaces[i].dirty).collect();
        self.run(memory, dirty)
    }

    /// makes guest memory right over a range something other than a draw
    /// is about to read or write, what the GPU drew there comes down, and
    /// the next draw looks at the memory again.
    pub(crate) fn sync<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        let overlapping: Vec<usize> = (0..self.surfaces.len()).filter(|&i| self.surfaces[i].overlaps(addr, len)).collect();
        let dirty: Vec<usize> = overlapping.iter().copied().filter(|&i| self.surfaces[i].dirty).collect();
        if !dirty.is_empty() {
            self.run(memory, dirty)?;
        }
        for i in overlapping {
            self.surfaces[i].checked = false;
        }
        Ok(())
    }

    /// runs what the batch recorded, then writes the given surfaces back to
    /// guest memory.
    fn run<M: GpuMemory>(&mut self, memory: &mut M, dirty: Vec<usize>) -> Result<(), String> {
        if !self.recording && dirty.is_empty() {
            return Ok(());
        }
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
        if !dirty.is_empty() {
            self.barrier();
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
                // SAFETY: recording, outside rendering, into a buffer big
                // enough for every region
                unsafe {
                    self.device.cmd_copy_image_to_buffer(self.commands, s.image.image, vk::ImageLayout::GENERAL, readback, &regions)
                };
                offsets.push(offset);
                offset += size_of(s);
            }
            let to_host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            // SAFETY: recording
            unsafe {
                self.device.cmd_pipeline_barrier2(self.commands, &vk::DependencyInfo::default().memory_barriers(&to_host))
            };
        }

        // SAFETY: the command buffer is recording and submitted once, then
        // waited for before anything touches what it used
        unsafe {
            self.device.end_command_buffer(self.commands).map_err(vk_error("end a command buffer"))?;
            let buffers = [self.commands];
            let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
            self.device.queue_submit(self.queue, &submit, self.fence).map_err(vk_error("submit"))?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).map_err(vk_error("wait for the GPU"))?;
            self.device.reset_fences(&[self.fence]).map_err(vk_error("reset a fence"))?;
            self.device
                .reset_command_buffer(self.commands, vk::CommandBufferResetFlags::empty())
                .map_err(vk_error("reset a command buffer"))?;
        }
        self.recording = false;
        if let Some(readback) = self.readback.as_ref().filter(|b| b.incoherent && !dirty.is_empty()) {
            let range = [vk::MappedMemoryRange::default().memory(readback.memory).offset(0).size(vk::WHOLE_SIZE)];
            // SAFETY: the memory is mapped and the GPU is done with it
            unsafe { self.device.invalidate_mapped_memory_ranges(&range) }.map_err(vk_error("invalidate memory"))?;
        }

        for (&i, &offset) in dirty.iter().zip(&offsets) {
            let (width, height, kind, addr) = {
                let s = &self.surfaces[i];
                (s.width, s.height, s.kind, s.addr)
            };
            let pixels = (width * height) as usize;
            let readback = self.readback.as_ref().expect("the readback buffer was sized above");
            // SAFETY: the GPU finished writing the buffer, and the range was
            // sized for this surface
            let data = unsafe { std::slice::from_raw_parts(readback.mapped.add(offset as usize), size_of(&self.surfaces[i]) as usize) };
            let mut bytes = vec![0u8; (pixels as u32 * kind.bytes()) as usize];
            match kind {
                Kind::Color(format) => {
                    let bpp = format.bytes_per_pixel();
                    for y in 0..height {
                        for x in 0..width {
                            let at = (((height - 1 - y) * width + x) * 4) as usize;
                            let rgba = [data[at], data[at + 1], data[at + 2], data[at + 3]];
                            let out = morton_offset(x, y, width, bpp as u32) as usize;
                            format.encode(rgba, &mut bytes[out..out + bpp]);
                        }
                    }
                }
                Kind::Depth(sample) => {
                    let (depths, stencils) = data.split_at(pixels * 4);
                    for y in 0..height {
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
            memory.write(addr, &bytes);
            let surface = &mut self.surfaces[i];
            surface.shadow = bytes;
            surface.dirty = false;
        }

        for surface in &mut self.surfaces {
            surface.checked = false;
        }
        self.used = 0;
        self.tables = None;
        self.batch += 1;

        // textures nobody drew with for a while go
        let batch = self.batch;
        let stale: Vec<usize> = self.textures.iter().filter(|(_, t)| batch - t.used > 600).map(|(&k, _)| k).collect();
        for key in stale {
            if let Some(texture) = self.textures.remove(&key) {
                self.destroy_image(&texture.image);
            }
        }
        Ok(())
    }
}

impl Drop for Hardware {
    fn drop(&mut self) {
        // SAFETY: waits for the GPU before anything it uses goes
        unsafe {
            let _ = self.device.device_wait_idle();
            for texture in self.textures.values() {
                self.destroy_image(&texture.image);
            }
            for surface in &self.surfaces {
                self.destroy_image(&surface.image);
            }
            self.destroy_image(&self.blank);
            for buffer in [Some(&self.ring), self.readback.as_ref()].into_iter().flatten() {
                self.device.destroy_buffer(buffer.buffer, None);
                self.device.free_memory(buffer.memory, None);
            }
            for &pipeline in self.pipelines.values() {
                self.device.destroy_pipeline(pipeline, None);
            }
            for &sampler in self.samplers.values() {
                self.device.destroy_sampler(sampler, None);
            }
            self.device.destroy_shader_module(self.vertex_shader, None);
            self.device.destroy_shader_module(self.fragment_shader, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// SAFETY: the mapped pointers belong to buffers only this value uses, and
// Vulkan handles may move between threads
unsafe impl Send for Hardware {}

/// the device to draw with and its graphics queue family, a discrete GPU
/// when there is one.
fn pick(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), String> {
    // SAFETY: queries on an instance we own
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(vk_error("list the GPUs"))?;
    let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
    for device in devices {
        // SAFETY: as above
        let (properties, features, depth, extensions, families) = unsafe {
            (
                instance.get_physical_device_properties(device),
                instance.get_physical_device_features(device),
                instance.get_physical_device_format_properties(device, DEPTH_FORMAT),
                instance.enumerate_device_extension_properties(device).unwrap_or_default(),
                instance.get_physical_device_queue_family_properties(device),
            )
        };
        let push = extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::push_descriptor::NAME));
        let usable = properties.api_version >= vk::API_VERSION_1_3
            && features.depth_clamp == vk::TRUE
            && depth.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            && push;
        let Some(family) = families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS)) else { continue };
        if !usable {
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
