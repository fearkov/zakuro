//! Vulkan presentation backend.

use std::ffi::CStr;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{layout, PresentError, Presenter, ScreenImage, Viewport};

const VERTEX_SPIRV: &[u8] = include_bytes!("../../shaders/present.vert.spv");
const FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present.frag.spv");

/// how many frames may be recorded before waiting on the oldest.
const FRAMES_IN_FLIGHT: usize = 2;

fn fail(message: impl Into<String>) -> PresentError {
    PresentError::Backend(message.into())
}

fn vk_fail(context: &str) -> impl Fn(vk::Result) -> PresentError + '_ {
    move |error| PresentError::Backend(format!("{context}: {error}"))
}

/// everything belonging to one emulated screen.
struct Screen {
    width: u32,
    height: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_mapping: *mut u8,
    staging_size: vk::DeviceSize,
    descriptor: vk::DescriptorSet,
    /// false until the image has been transitioned out of UNDEFINED.
    initialized: bool,
    /// set when new pixels were written into the staging buffer this frame.
    dirty: bool,
}

pub struct VulkanPresenter {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_instance: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family: u32,
    queue: vk::Queue,

    swapchain_device: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame: usize,

    screens: [Screen; 2],
    window: (u32, u32),
    /// set when the swapchain no longer matches the window.
    stale: bool,
}

impl VulkanPresenter {
    pub fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        size: (u32, u32),
    ) -> Result<VulkanPresenter, PresentError> {
        // SAFETY: the loader reads the system's Vulkan library, which is the
        // documented way to start.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| fail(format!("no Vulkan loader: {e}")))?;

        let display = window
            .display_handle()
            .map_err(|e| fail(format!("no display handle: {e}")))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| fail(format!("no window handle: {e}")))?;

        let application = vk::ApplicationInfo::default()
            .application_name(c"Zakuro")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Zakuro")
            .api_version(vk::API_VERSION_1_1);

        let extensions = ash_window::enumerate_required_extensions(display.as_raw())
            .map_err(vk_fail("querying surface extensions"))?
            .to_vec();

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(vk_fail("creating the instance"))?;

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(vk_fail("creating the surface"))?;

        let (physical_device, queue_family) =
            pick_device(&instance, &surface_instance, surface)?;

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        log::info!("Vulkan on {}", name.to_string_lossy());

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(vk_fail("creating the device"))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);
        let surface_format = choose_format(&surface_instance, physical_device, surface)?;
        let render_pass = create_render_pass(&device, surface_format.format)?;

        let descriptor_layout = create_descriptor_layout(&device)?;
        let (pipeline_layout, pipeline) =
            create_pipeline(&device, render_pass, descriptor_layout)?;

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    // nearest keeps the console's pixels crisp when scaled up.
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(vk_fail("creating the sampler"))?;

        let descriptor_pool = unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2)];
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(2),
                None,
            )
        }
        .map_err(vk_fail("creating the descriptor pool"))?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(vk_fail("creating the command pool"))?;

        let command_buffers = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )
        }
        .map_err(vk_fail("allocating command buffers"))?;

        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            image_available.push(
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .map_err(vk_fail("creating a semaphore"))?,
            );
            render_finished.push(
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .map_err(vk_fail("creating a semaphore"))?,
            );
            in_flight.push(
                unsafe {
                    device.create_fence(
                        // signalled, so the first frame does not wait forever.
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                }
                .map_err(vk_fail("creating a fence"))?,
            );
        }

        // the two screens start at their native sizes, upload grows them if
        // a title ever reports something different.
        let screens = [
            Screen::new(
                &device,
                &memory_properties,
                descriptor_pool,
                descriptor_layout,
                sampler,
                400,
                240,
            )?,
            Screen::new(
                &device,
                &memory_properties,
                descriptor_pool,
                descriptor_layout,
                sampler,
                320,
                240,
            )?,
        ];

        let mut presenter = VulkanPresenter {
            _entry: entry,
            instance,
            surface_instance,
            surface,
            physical_device,
            memory_properties,
            device,
            queue_family,
            queue,
            swapchain_device,
            swapchain: vk::SwapchainKHR::null(),
            surface_format,
            extent: vk::Extent2D {
                width: size.0,
                height: size.1,
            },
            image_views: Vec::new(),
            framebuffers: Vec::new(),
            render_pass,
            descriptor_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            sampler,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame: 0,
            screens,
            window: size,
            stale: false,
        };

        presenter.build_swapchain()?;
        Ok(presenter)
    }

    fn build_swapchain(&mut self) -> Result<(), PresentError> {
        let capabilities = unsafe {
            self.surface_instance
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(vk_fail("querying surface capabilities"))?;

        // a current extent of u32::MAX means the surface takes its size from
        // the swapchain rather than the other way round.
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: self.window.0.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: self.window.1.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };

        if extent.width == 0 || extent.height == 0 {
            // the window is minimized, nothing to build.
            self.extent = extent;
            return Ok(());
        }

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let present_modes = unsafe {
            self.surface_instance
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
        .map_err(vk_fail("querying present modes"))?;
        // FIFO is always available and matches the console's own vsync.
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            present_modes[0]
        };

        let old = self.swapchain;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        let swapchain = unsafe { self.swapchain_device.create_swapchain(&info, None) }
            .map_err(vk_fail("creating the swapchain"))?;

        self.destroy_swapchain_resources();
        if old != vk::SwapchainKHR::null() {
            unsafe { self.swapchain_device.destroy_swapchain(old, None) };
        }

        self.swapchain = swapchain;
        self.extent = extent;

        let images = unsafe { self.swapchain_device.get_swapchain_images(swapchain) }
            .map_err(vk_fail("getting swapchain images"))?;

        for image in images {
            let view = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.surface_format.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }
            .map_err(vk_fail("creating a swapchain image view"))?;

            let attachments = [view];
            let framebuffer = unsafe {
                self.device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(vk_fail("creating a framebuffer"))?;

            self.image_views.push(view);
            self.framebuffers.push(framebuffer);
        }

        self.stale = false;
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
        }
    }

    /// copies pixels into a screen's staging buffer, reallocating the image if
    /// the size changed.
    fn upload(&mut self, index: usize, image: &ScreenImage<'_>) -> Result<(), PresentError> {
        if self.screens[index].width != image.width || self.screens[index].height != image.height
        {
            unsafe { self.device.device_wait_idle() }.map_err(vk_fail("waiting for idle"))?;
            let old = std::mem::replace(
                &mut self.screens[index],
                Screen::new(
                    &self.device,
                    &self.memory_properties,
                    self.descriptor_pool,
                    self.descriptor_layout,
                    self.sampler,
                    image.width,
                    image.height,
                )?,
            );
            old.destroy(&self.device);
        }

        let screen = &mut self.screens[index];
        let bytes = (image.width * image.height * 4) as usize;
        let count = bytes.min(image.pixels.len());
        // SAFETY: the staging buffer is host visible, persistently mapped and
        // at least bytes long, and no command using it is in flight because
        // the frame's fence was waited on first.
        unsafe {
            std::ptr::copy_nonoverlapping(image.pixels.as_ptr(), screen.staging_mapping, count);
        }
        screen.dirty = true;
        Ok(())
    }

    fn record(&self, command_buffer: vk::CommandBuffer, framebuffer: vk::Framebuffer) {
        let device = &self.device;
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin command buffer");

            for screen in &self.screens {
                if !screen.dirty {
                    continue;
                }
                let old_layout = if screen.initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                };
                transition(
                    device,
                    command_buffer,
                    screen.image,
                    old_layout,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    screen.staging,
                    screen.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: screen.width,
                            height: screen.height,
                            depth: 1,
                        })],
                );

                transition(
                    device,
                    command_buffer,
                    screen.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }

            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let (top, bottom) = layout(self.extent.width, self.extent.height);
            for (screen, viewport) in self.screens.iter().zip([top, bottom]) {
                if !screen.initialized && !screen.dirty {
                    continue;
                }
                set_viewport(device, command_buffer, viewport, self.extent);
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[screen.descriptor],
                    &[],
                );
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
            }

            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer).expect("end command buffer");
        }
    }
}

impl Presenter for VulkanPresenter {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
    ) -> Result<(), PresentError> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }
        if self.stale {
            unsafe { self.device.device_wait_idle() }.map_err(vk_fail("waiting for idle"))?;
            self.build_swapchain()?;
            return Err(PresentError::OutOfDate);
        }

        let frame = self.frame;
        let fence = self.in_flight[frame];
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(vk_fail("waiting for a fence"))?;
        }

        let acquired = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            )
        };
        let (image_index, suboptimal) = match acquired {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stale = true;
                return Err(PresentError::OutOfDate);
            }
            Err(error) => return Err(vk_fail("acquiring an image")(error)),
        };

        // only reset the fence once we know we are going to submit.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(vk_fail("resetting a fence"))?;
        }

        for screen in &mut self.screens {
            screen.dirty = false;
        }
        if !top.is_empty() {
            self.upload(0, &top)?;
        }
        if !bottom.is_empty() {
            self.upload(1, &bottom)?;
        }

        let command_buffer = self.command_buffers[frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(vk_fail("resetting a command buffer"))?;
        }
        self.record(command_buffer, self.framebuffers[image_index as usize]);

        for screen in &mut self.screens {
            if screen.dirty {
                screen.initialized = true;
            }
        }

        let wait = [self.image_available[frame]];
        let signal = [self.render_finished[frame]];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let buffers = [command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(vk_fail("submitting"))?;
        }

        let swapchains = [self.swapchain];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let result = unsafe { self.swapchain_device.queue_present(self.queue, &present) };

        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;

        match result {
            Ok(false) if !suboptimal => Ok(()),
            Ok(_) => {
                self.stale = true;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stale = true;
                Err(PresentError::OutOfDate)
            }
            Err(error) => Err(vk_fail("presenting")(error)),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window = (width.max(1), height.max(1));
        self.stale = true;
    }
}

impl Drop for VulkanPresenter {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for screen in std::mem::replace(&mut self.screens, [Screen::null(), Screen::null()]) {
                screen.destroy(&self.device);
            }
            self.destroy_swapchain_resources();
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_device.destroy_swapchain(self.swapchain, None);
            }
            for semaphore in self.image_available.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
        let _ = self.queue_family;
    }
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

fn pick_device(
    instance: &ash::Instance,
    surface_instance: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), PresentError> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_fail("enumerating devices"))?;

    let mut best: Option<(vk::PhysicalDevice, u32, u32)> = None;
    for device in devices {
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in families.iter().enumerate() {
            if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            let supported = unsafe {
                surface_instance.get_physical_device_surface_support(
                    device,
                    index as u32,
                    surface,
                )
            }
            .unwrap_or(false);
            if !supported {
                continue;
            }
            // prefer a discrete GPU, then integrated, then anything.
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            };
            if best.is_none_or(|(_, _, current)| score > current) {
                best = Some((device, index as u32, score));
            }
            break;
        }
    }

    best.map(|(device, family, _)| (device, family))
        .ok_or_else(|| fail("no Vulkan device can present to this window"))
}

fn choose_format(
    surface_instance: &ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::SurfaceFormatKHR, PresentError> {
    let formats = unsafe {
        surface_instance.get_physical_device_surface_formats(physical_device, surface)
    }
    .map_err(vk_fail("querying surface formats"))?;

    // the emulator produces straight sRGB-ish bytes, so an UNORM target keeps
    // them looking the way the console does.
    formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
        .ok_or_else(|| fail("the surface reports no formats"))
}

fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, PresentError> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];

    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
    .map_err(vk_fail("creating the render pass"))
}

fn create_descriptor_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, PresentError> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(vk_fail("creating the descriptor set layout"))
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline), PresentError> {
    let vertex = create_shader_module(device, VERTEX_SPIRV)?;
    let fragment = create_shader_module(device, FRAGMENT_SPIRV)?;

    let layouts = [descriptor_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
            None,
        )
    }
    .map_err(vk_fail("creating the pipeline layout"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];

    // the vertex shader builds its own triangle, so there is no input state.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

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

    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, error)| vk_fail("creating the pipeline")(error))?;

    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }

    Ok((pipeline_layout, pipelines[0]))
}

fn create_shader_module(
    device: &ash::Device,
    spirv: &[u8],
) -> Result<vk::ShaderModule, PresentError> {
    if !spirv.len().is_multiple_of(4) {
        return Err(fail("SPIR-V is not a whole number of words"));
    }
    let words: Vec<u32> = spirv
        .as_chunks::<4>().0.iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&words),
            None,
        )
    }
    .map_err(vk_fail("creating a shader module"))
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && properties.memory_types[i as usize]
                .property_flags
                .contains(wanted)
    })
}

fn transition(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match (from, to) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn set_viewport(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    viewport: Viewport,
    extent: vk::Extent2D,
) {
    unsafe {
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: viewport.x,
                y: viewport.y,
                width: viewport.width,
                height: viewport.height,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        // the scissor has to be clamped to the framebuffer or validation
        // complains when the window is smaller than the layout wants.
        let x = viewport.x.max(0.0) as i32;
        let y = viewport.y.max(0.0) as i32;
        let width = (viewport.width as u32).min(extent.width.saturating_sub(x as u32));
        let height = (viewport.height as u32).min(extent.height.saturating_sub(y as u32));
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D { width, height },
            }],
        );
    }
}

impl Screen {
    fn null() -> Screen {
        Screen {
            width: 0,
            height: 0,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            staging: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_mapping: std::ptr::null_mut(),
            staging_size: 0,
            descriptor: vk::DescriptorSet::null(),
            initialized: false,
            dirty: false,
        }
    }

    fn new(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
        width: u32,
        height: u32,
    ) -> Result<Screen, PresentError> {
        let size = (width as vk::DeviceSize) * (height as vk::DeviceSize) * 4;

        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(vk_fail("creating a screen image"))?;

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let type_index = find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| fail("no device-local memory for a screen image"))?;
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(type_index),
                None,
            )
        }
        .map_err(vk_fail("allocating image memory"))?;
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(vk_fail("binding image memory"))?;

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(vk_fail("creating a screen image view"))?;

        let staging = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(vk_fail("creating a staging buffer"))?;

        let staging_requirements = unsafe { device.get_buffer_memory_requirements(staging) };
        let staging_type = find_memory_type(
            memory_properties,
            staging_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| fail("no host-visible memory for a staging buffer"))?;
        let staging_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_requirements.size)
                    .memory_type_index(staging_type),
                None,
            )
        }
        .map_err(vk_fail("allocating staging memory"))?;
        unsafe { device.bind_buffer_memory(staging, staging_memory, 0) }
            .map_err(vk_fail("binding staging memory"))?;

        // mapped once and kept mapped, the buffer is written every frame.
        let staging_mapping = unsafe {
            device.map_memory(
                staging_memory,
                0,
                staging_requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(vk_fail("mapping staging memory"))? as *mut u8;

        let layouts = [layout];
        let descriptors = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(vk_fail("allocating a descriptor set"))?;
        let descriptor = descriptors[0];

        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)],
                &[],
            );
        }

        Ok(Screen {
            width,
            height,
            image,
            memory,
            view,
            staging,
            staging_memory,
            staging_mapping,
            staging_size: staging_requirements.size,
            descriptor,
            initialized: false,
            dirty: false,
        })
    }

    fn destroy(self, device: &ash::Device) {
        if self.image == vk::Image::null() {
            return;
        }
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
            device.unmap_memory(self.staging_memory);
            device.destroy_buffer(self.staging, None);
            device.free_memory(self.staging_memory, None);
        }
        let _ = self.staging_size;
    }
}
