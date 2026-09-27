//! the overlay, drawn with Vulkan over the screens in the same render pass.

use std::collections::HashMap;

use ash::vk;

use super::vulkan::{create_shader_module, fail, find_memory_type, transition, vk_fail};
use super::{Overlay, OverlayTexture, OverlayVertex, PresentError};

const VERTEX_SPIRV: &[u8] = include_bytes!("../../shaders/overlay.vert.spv");
const FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/overlay.frag.spv");

/// textures the overlay may have at once.
const MAX_TEXTURES: u32 = 256;

/// what the shaders take besides the triangles.
#[repr(C)]
struct Push {
    size: [f32; 2],
    srgb: u32,
    _pad: u32,
}

struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    descriptor: vk::DescriptorSet,
    /// never drawn into yet, so its contents need not be kept.
    fresh: bool,
}

/// memory the CPU writes and the GPU reads, mapped for good.
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
}

/// what may only go once the GPU is done with the frame that used it.
enum Garbage {
    Texture(Texture),
    Buffer(Buffer),
}

pub(super) struct OverlayPainter {
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// nearest, then linear.
    samplers: [vk::Sampler; 2],
    textures: HashMap<u64, Texture>,
    /// each frame in flight's vertices and indices.
    geometry: Vec<Option<Buffer>>,
    /// what each frame in flight leaves to destroy once it is done.
    garbage: Vec<Vec<Garbage>>,
    /// where the indices start in this frame's buffer.
    index_offset: u64,
    /// texture changes and frees from frames that were never drawn, which
    /// still have to happen, the overlay sends each only once.
    pending: Vec<OverlayTexture>,
    pending_free: Vec<u64>,
    srgb: bool,
}

impl OverlayPainter {
    pub(super) fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        frames: usize,
        srgb: bool,
    ) -> Result<OverlayPainter, PresentError> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
        }
        .map_err(vk_fail("creating the overlay descriptor layout"))?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_TEXTURES)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&sizes)
                    .max_sets(MAX_TEXTURES),
                None,
            )
        }
        .map_err(vk_fail("creating the overlay descriptor pool"))?;

        let sampler = |filter: vk::Filter| unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(filter)
                    .min_filter(filter)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        };
        let samplers = [
            sampler(vk::Filter::NEAREST).map_err(vk_fail("creating an overlay sampler"))?,
            sampler(vk::Filter::LINEAR).map_err(vk_fail("creating an overlay sampler"))?,
        ];

        let (pipeline_layout, pipeline) = create_pipeline(device, render_pass, descriptor_layout)?;
        Ok(OverlayPainter {
            descriptor_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            samplers,
            textures: HashMap::new(),
            geometry: (0..frames).map(|_| None).collect(),
            garbage: (0..frames).map(|_| Vec::new()).collect(),
            index_offset: 0,
            pending: Vec::new(),
            pending_free: Vec::new(),
            srgb,
        })
    }

    /// keeps what an overlay changes for the next frame drawn, when this one
    /// is not.
    pub(super) fn defer(&mut self, overlay: &Overlay) {
        self.pending.extend(overlay.textures.iter().map(|texture| OverlayTexture {
            id: texture.id,
            offset: texture.offset,
            size: texture.size,
            pixels: texture.pixels.clone(),
            linear: texture.linear,
        }));
        self.pending_free.extend_from_slice(&overlay.free);
    }

    /// what the frame that last used this slot left behind can go, the
    /// caller having waited for it.
    pub(super) fn begin_frame(&mut self, device: &ash::Device, frame: usize) {
        for garbage in self.garbage[frame].drain(..) {
            destroy(device, self.descriptor_pool, garbage);
        }
    }

    /// records the texture changes and fills this frame's geometry, before
    /// the render pass starts.
    pub(super) fn upload(
        &mut self,
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        command_buffer: vk::CommandBuffer,
        frame: usize,
        overlay: &Overlay,
    ) -> Result<(), PresentError> {
        let pending = std::mem::take(&mut self.pending);
        for texture in pending.iter().chain(&overlay.textures) {
            let [width, height] = texture.size;
            if width == 0 || height == 0 || texture.pixels.len() < (width * height * 4) as usize {
                continue;
            }
            if texture.offset.is_none() {
                let made = self.make_texture(device, memory, texture.size, texture.linear)?;
                if let Some(old) = self.textures.insert(texture.id, made) {
                    self.garbage[frame].push(Garbage::Texture(old));
                }
            }
            let Some(target) = self.textures.get_mut(&texture.id) else { continue };
            let staging = make_buffer(device, memory, texture.pixels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
            // SAFETY: the buffer is mapped and as long as the pixels
            unsafe { std::ptr::copy_nonoverlapping(texture.pixels.as_ptr(), staging.mapped, texture.pixels.len()) };
            let [x, y] = texture.offset.unwrap_or([0, 0]);
            let from = if target.fresh { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
            transition(device, command_buffer, target.image, from, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            // SAFETY: recording, outside a render pass, into an image in the
            // layout the copy wants
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging.buffer,
                    target.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1),
                        )
                        .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                        .image_extent(vk::Extent3D { width, height, depth: 1 })],
                );
            }
            transition(
                device,
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            target.fresh = false;
            self.garbage[frame].push(Garbage::Buffer(staging));
        }

        // the vertices, then the indices
        let vertices: usize = overlay.meshes.iter().map(|m| m.vertices.len()).sum();
        let indices: usize = overlay.meshes.iter().map(|m| m.indices.len()).sum();
        let vertex_bytes = (vertices * size_of::<OverlayVertex>()) as u64;
        self.index_offset = vertex_bytes.next_multiple_of(4);
        let needed = self.index_offset + (indices * 4) as u64;
        if needed == 0 {
            return Ok(());
        }
        if self.geometry[frame].as_ref().is_none_or(|buffer| buffer.size < needed) {
            let usage = vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER;
            let grown = make_buffer(device, memory, needed.next_power_of_two().max(64 * 1024), usage)?;
            if let Some(old) = self.geometry[frame].replace(grown) {
                self.garbage[frame].push(Garbage::Buffer(old));
            }
        }
        let buffer = self.geometry[frame].as_ref().expect("sized above");
        let (mut vertex_at, mut index_at) = (0usize, self.index_offset as usize);
        for mesh in &overlay.meshes {
            let vertex_bytes = mesh.vertices.len() * size_of::<OverlayVertex>();
            // SAFETY: the buffer holds every mesh's vertices and indices,
            // and the vertex type is plain data
            unsafe {
                std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr() as *const u8, buffer.mapped.add(vertex_at), vertex_bytes);
                std::ptr::copy_nonoverlapping(mesh.indices.as_ptr() as *const u8, buffer.mapped.add(index_at), mesh.indices.len() * 4);
            }
            vertex_at += vertex_bytes;
            index_at += mesh.indices.len() * 4;
        }
        Ok(())
    }

    /// records the draws, inside the render pass.
    pub(super) fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer, frame: usize, overlay: &Overlay, extent: vk::Extent2D) {
        let Some(buffer) = &self.geometry[frame] else { return };
        if overlay.meshes.is_empty() || extent.width == 0 || extent.height == 0 {
            return;
        }
        let push = Push { size: [extent.width as f32, extent.height as f32], srgb: self.srgb as u32, _pad: 0 };
        // SAFETY: recording inside the render pass the pipeline was made for,
        // with buffers the upload filled for this frame
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(&push as *const Push as *const u8, size_of::<Push>()),
            );
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(command_buffer, buffer.buffer, self.index_offset, vk::IndexType::UINT32);
        }
        let (mut first_vertex, mut first_index) = (0i32, 0u32);
        for mesh in &overlay.meshes {
            let count = mesh.indices.len() as u32;
            let [left, top, right, bottom] = mesh.clip;
            let (right, bottom) = (right.min(extent.width), bottom.min(extent.height));
            if let (Some(texture), true) = (self.textures.get(&mesh.texture), left < right && top < bottom && count > 0) {
                // SAFETY: as above
                unsafe {
                    device.cmd_set_scissor(
                        command_buffer,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: left as i32, y: top as i32 },
                            extent: vk::Extent2D { width: right - left, height: bottom - top },
                        }],
                    );
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[texture.descriptor],
                        &[],
                    );
                    device.cmd_draw_indexed(command_buffer, count, 1, first_index, first_vertex, 0);
                }
            }
            first_vertex += mesh.vertices.len() as i32;
            first_index += count;
        }
    }

    /// textures the overlay no longer needs, gone once this frame is done.
    pub(super) fn free(&mut self, frame: usize, overlay: &Overlay) {
        let pending = std::mem::take(&mut self.pending_free);
        for id in pending.iter().chain(&overlay.free) {
            if let Some(texture) = self.textures.remove(id) {
                self.garbage[frame].push(Garbage::Texture(texture));
            }
        }
    }

    fn make_texture(
        &self,
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        [width, height]: [u32; 2],
        linear: bool,
    ) -> Result<Texture, PresentError> {
        let format = vk::Format::R8G8B8A8_UNORM;
        // SAFETY: plain object creation on a live device
        unsafe {
            let image = device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(vk_fail("creating an overlay texture"))?;
            let requirements = device.get_image_memory_requirements(image);
            let type_index = find_memory_type(memory, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or_else(|| fail("no device memory for an overlay texture"))?;
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(type_index),
                    None,
                )
                .map_err(vk_fail("allocating an overlay texture"))?;
            device.bind_image_memory(image, memory, 0).map_err(vk_fail("binding an overlay texture"))?;
            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
                .map_err(vk_fail("creating an overlay texture view"))?;
            let layouts = [self.descriptor_layout];
            let descriptor = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptor_pool).set_layouts(&layouts),
                )
                .map_err(vk_fail("allocating an overlay descriptor"))?[0];
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.samplers[linear as usize])
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&info)],
                &[],
            );
            Ok(Texture { image, memory, view, descriptor, fresh: true })
        }
    }

    pub(super) fn destroy(&mut self, device: &ash::Device) {
        let pool = self.descriptor_pool;
        for garbage in self.garbage.iter_mut().flat_map(|g| g.drain(..)) {
            destroy(device, pool, garbage);
        }
        for buffer in self.geometry.iter_mut().filter_map(Option::take) {
            destroy(device, pool, Garbage::Buffer(buffer));
        }
        for (_, texture) in self.textures.drain() {
            destroy(device, pool, Garbage::Texture(texture));
        }
        // SAFETY: the device is idle, the caller waited for it
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            for sampler in self.samplers {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

fn destroy(device: &ash::Device, pool: vk::DescriptorPool, garbage: Garbage) {
    // SAFETY: the GPU finished every frame that used it
    unsafe {
        match garbage {
            Garbage::Texture(texture) => {
                device.destroy_image_view(texture.view, None);
                device.destroy_image(texture.image, None);
                device.free_memory(texture.memory, None);
                // the pool lets sets go one by one, and a failure only leaks one
                let _ = device.free_descriptor_sets(pool, &[texture.descriptor]);
            }
            Garbage::Buffer(buffer) => {
                device.unmap_memory(buffer.memory);
                device.destroy_buffer(buffer.buffer, None);
                device.free_memory(buffer.memory, None);
            }
        }
    }
}

fn make_buffer(
    device: &ash::Device,
    memory: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<Buffer, PresentError> {
    // SAFETY: plain object creation on a live device
    unsafe {
        let buffer = device
            .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)
            .map_err(vk_fail("creating an overlay buffer"))?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let type_index = find_memory_type(
            memory,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| fail("no host memory for an overlay buffer"))?;
        let allocation = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(type_index),
                None,
            )
            .map_err(vk_fail("allocating an overlay buffer"))?;
        device.bind_buffer_memory(buffer, allocation, 0).map_err(vk_fail("binding an overlay buffer"))?;
        let mapped = device
            .map_memory(allocation, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(vk_fail("mapping an overlay buffer"))? as *mut u8;
        Ok(Buffer { buffer, memory: allocation, size, mapped })
    }
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline), PresentError> {
    let vertex = create_shader_module(device, VERTEX_SPIRV)?;
    let fragment = create_shader_module(device, FRAGMENT_SPIRV)?;

    let layouts = [descriptor_layout];
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .size(size_of::<Push>() as u32)];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
            None,
        )
    }
    .map_err(vk_fail("creating the overlay pipeline layout"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vertex).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fragment).name(c"main"),
    ];
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<OverlayVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attributes = [
        vk::VertexInputAttributeDescription::default().location(0).binding(0).format(vk::Format::R32G32_SFLOAT).offset(0),
        vk::VertexInputAttributeDescription::default().location(1).binding(0).format(vk::Format::R32G32_SFLOAT).offset(8),
        vk::VertexInputAttributeDescription::default().location(2).binding(0).format(vk::Format::R8G8B8A8_UNORM).offset(16),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // premultiplied alpha over what is already drawn
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
        .map_err(|(_, error)| vk_fail("creating the overlay pipeline")(error))?;
    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }
    Ok((pipeline_layout, pipelines[0]))
}
