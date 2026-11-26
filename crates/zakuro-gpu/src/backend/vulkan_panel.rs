//! the bottom screen on a second panel, the one a compositor leases through drm-lease-v1 (a dual-screen handheld's
//! lower panel in gamescope), drawn on the presenter's device, so the renderer's picture of the screen gets drawn
//! straight from its image
//!
//! create_surface makes the calls of CreateDrmDisplaySurface in drm_display_surface.cpp of MHGU Recomp's AArch64 Linux
//! builds (https://github.com/cerroverb/MHGURecomp-aarch64 commit 8c10fd6, third_party/azahar-drm-lease), Azahar's
//! code (https://github.com/virtudude/azahar branch drm-leasing commit d16299140,
//! src/video_core/renderer_vulkan/vk_platform.cpp, GPLv2 or later), in the same order, through ash
//! the swapchain gets the panel's fewest images, as in msr2ds's panel thread
//! (https://github.com/gminustwo/msr2ds/pull/94, runtime/second_display_lease.c), so a picture waits a refresh at most

use std::ffi::c_char;
use std::sync::mpsc;

use ash::vk;

use super::drm_lease::{self, Output};
use super::vulkan::{choose_format, create_pipeline, create_render_pass, fail, set_viewport, vk_fail, CONSTANTS, FRAGMENT_SPIRV, FRAG_CONSTANTS_OFFSET, FRAMES_IN_FLIGHT, VERT_CONSTANTS_OFFSET};
use super::{layout, PresentError, ScreenLayout, ScreenOrientation};

pub(super) struct Panel {
    display_instance: ash::khr::display::Instance,
    direct_mode: ash::ext::direct_mode_display::Instance,
    display: vk::DisplayKHR,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    /// per frame in flight, signaled when the panel's image is free to draw in
    available: Vec<vk::Semaphore>,
    /// per frame in flight, signaled when the picture's drawn
    finished: Vec<vk::Semaphore>,
    /// the way the screen's drawn to stand upright on the panel
    orientation: ScreenOrientation,
}

/// what the presenter's own setup hands the panel
pub(super) struct Device<'a> {
    pub entry: &'a ash::Entry,
    pub instance: &'a ash::Instance,
    pub surface_instance: &'a ash::khr::surface::Instance,
    pub swapchain_device: &'a ash::khr::swapchain::Device,
    pub physical: vk::PhysicalDevice,
    pub queue_family: u32,
    pub device: &'a ash::Device,
    pub descriptor_layout: vk::DescriptorSetLayout,
}

impl Panel {
    /// the panel ready to draw on, everything made on the way gets destroyed again on a failure
    pub(super) fn new(device: &Device, output: &Output) -> Result<Panel, PresentError> {
        let mut panel = Panel {
            display_instance: ash::khr::display::Instance::new(device.entry, device.instance),
            direct_mode: ash::ext::direct_mode_display::Instance::new(device.entry, device.instance),
            display: vk::DisplayKHR::null(),
            surface: vk::SurfaceKHR::null(),
            swapchain: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            render_pass: vk::RenderPass::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            image_views: Vec::new(),
            framebuffers: Vec::new(),
            available: Vec::new(),
            finished: Vec::new(),
            orientation: ScreenOrientation::for_panel_rotation(output.rotation),
        };
        match panel.build(device, output) {
            Ok(()) => Ok(panel),
            Err(error) => {
                panel.destroy(device);
                Err(error)
            }
        }
    }

    fn build(&mut self, device: &Device, output: &Output) -> Result<(), PresentError> {
        let (surface, mode) = self.create_surface(device, output)?;
        self.surface = surface;
        // safe bc it's a query of the device's queue family and the surface just made
        let supported = unsafe { device.surface_instance.get_physical_device_surface_support(device.physical, device.queue_family, surface) }
            .map_err(vk_fail("querying the second display's support"))?;
        if !supported {
            return Err(fail("the queue that draws the screens can't present to the second display"));
        }
        let format = choose_format(device.surface_instance, device.physical, surface)?;
        self.render_pass = create_render_pass(device.device, format.format)?;
        (self.pipeline_layout, self.pipeline) = create_pipeline(device.device, self.render_pass, device.descriptor_layout, FRAGMENT_SPIRV, CONSTANTS)?;
        self.create_swapchain(device, format, mode)?;
        for _ in 0..FRAMES_IN_FLIGHT {
            let info = vk::SemaphoreCreateInfo::default();
            // safe bc the device is alive and the semaphores are destroyed w/ the panel
            unsafe {
                self.available.push(device.device.create_semaphore(&info, None).map_err(vk_fail("creating a semaphore"))?);
                self.finished.push(device.device.create_semaphore(&info, None).map_err(vk_fail("creating a semaphore"))?);
            }
        }
        log::info!(
            "the bottom screen shows on the second display, connector {} at {}x{}, turned {} degrees",
            output.connector_id,
            self.extent.width,
            self.extent.height,
            output.rotation
        );
        Ok(())
    }

    /// a VK_KHR_display surface on the leased connector, and the mode it shows in
    fn create_surface(&mut self, device: &Device, output: &Output) -> Result<(vk::SurfaceKHR, vk::DisplayModeParametersKHR), PresentError> {
        let acquire_drm = ash::ext::acquire_drm_display::Instance::new(device.entry, device.instance);
        let display_instance = &self.display_instance;
        let physical = device.physical;
        // safe bc these are queries and calls on the instance's GPU w/ the lease's own fd and connector
        unsafe {
            let display = acquire_drm
                .get_drm_display(physical, output.fd, output.connector_id)
                .map_err(vk_fail("vkGetDrmDisplayEXT"))?;
            acquire_drm.acquire_drm_display(physical, output.fd, display).map_err(vk_fail("vkAcquireDrmDisplayEXT"))?;
            self.display = display;

            // Vulkan doesn't order the modes, so the one the lease reported gets matched
            let modes = display_instance.get_display_mode_properties(physical, display).map_err(vk_fail("listing the second display's modes"))?;
            let mode = modes
                .iter()
                .find(|mode| mode.parameters.visible_region.width == output.width && mode.parameters.visible_region.height == output.height)
                .ok_or_else(|| fail(format!("the second display has no {}x{} mode", output.width, output.height)))?;

            let displays = display_instance.get_physical_device_display_properties(physical).map_err(vk_fail("listing displays"))?;
            let identity = displays
                .iter()
                .find(|properties| properties.display == display)
                .is_some_and(|properties| properties.supported_transforms.contains(vk::SurfaceTransformFlagsKHR::IDENTITY));
            if !identity {
                return Err(fail("the second display does not support the identity transform"));
            }

            let planes = display_instance.get_physical_device_display_plane_properties(physical).map_err(vk_fail("listing display planes"))?;
            let mut plane_index = None;
            for (index, plane) in planes.iter().enumerate() {
                if plane.current_display != vk::DisplayKHR::null() && plane.current_display != display {
                    continue;
                }
                let supported = display_instance.get_display_plane_supported_displays(physical, index as u32).map_err(vk_fail("listing a plane's displays"))?;
                if supported.contains(&display) {
                    plane_index = Some(index as u32);
                    break;
                }
            }
            let plane_index = plane_index.ok_or_else(|| fail("no display plane supports the second display"))?;

            let capabilities = display_instance
                .get_display_plane_capabilities(physical, mode.display_mode, plane_index)
                .map_err(vk_fail("querying the plane's capabilities"))?;
            let alpha = [
                vk::DisplayPlaneAlphaFlagsKHR::OPAQUE,
                vk::DisplayPlaneAlphaFlagsKHR::GLOBAL,
                vk::DisplayPlaneAlphaFlagsKHR::PER_PIXEL,
                vk::DisplayPlaneAlphaFlagsKHR::PER_PIXEL_PREMULTIPLIED,
            ]
            .into_iter()
            .find(|candidate| capabilities.supported_alpha.contains(*candidate))
            .ok_or_else(|| fail("the second display's plane advertises no alpha mode"))?;

            let info = vk::DisplaySurfaceCreateInfoKHR::default()
                .display_mode(mode.display_mode)
                .plane_index(plane_index)
                .plane_stack_index(planes[plane_index as usize].current_stack_index)
                .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .global_alpha(1.0)
                .alpha_mode(alpha)
                .image_extent(mode.parameters.visible_region);
            let surface = display_instance.create_display_plane_surface(&info, None).map_err(vk_fail("vkCreateDisplayPlaneSurfaceKHR"))?;
            Ok((surface, mode.parameters))
        }
    }

    fn create_swapchain(&mut self, device: &Device, format: vk::SurfaceFormatKHR, mode: vk::DisplayModeParametersKHR) -> Result<(), PresentError> {
        let surface_instance = device.surface_instance;
        // safe bc these are queries and objects on the surface and device, destroyed w/ the panel
        unsafe {
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(device.physical, self.surface)
                .map_err(vk_fail("querying the second display's capabilities"))?;
            self.extent = if capabilities.current_extent.width == u32::MAX { mode.visible_region } else { capabilities.current_extent };
            let modes = surface_instance
                .get_physical_device_surface_present_modes(device.physical, self.surface)
                .map_err(vk_fail("querying the second display's present modes"))?;
            // the newest picture replaces a waiting one where the panel offers that, FIFO's always there
            let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) { vk::PresentModeKHR::MAILBOX } else { vk::PresentModeKHR::FIFO };
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(capabilities.min_image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(self.extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);
            self.swapchain = device.swapchain_device.create_swapchain(&info, None).map_err(vk_fail("creating the second display's swapchain"))?;
            let images = device.swapchain_device.get_swapchain_images(self.swapchain).map_err(vk_fail("getting the second display's images"))?;
            let mode_name = if present_mode == vk::PresentModeKHR::MAILBOX { "mailbox" } else { "fifo" };
            log::info!("the second display presents in {mode_name} mode w/ {} images", images.len());
            for image in images {
                let view = device
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)),
                        None,
                    )
                    .map_err(vk_fail("creating a second display image view"))?;
                self.image_views.push(view);
                let attachments = [view];
                let framebuffer = device
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(self.render_pass)
                            .attachments(&attachments)
                            .width(self.extent.width)
                            .height(self.extent.height)
                            .layers(1),
                        None,
                    )
                    .map_err(vk_fail("creating a second display framebuffer"))?;
                self.framebuffers.push(framebuffer);
            }
        }
        Ok(())
    }

    /// an image to draw this frame's picture in, w/o waiting, None while the panel still shows or waits to show the
    /// ones it has, then the frame's picture goes unshown there
    pub(super) fn acquire(&self, swapchain_device: &ash::khr::swapchain::Device, frame: usize) -> Result<Option<u32>, vk::Result> {
        // safe bc the frame's fence was waited on, so its semaphore has no wait pending
        match unsafe { swapchain_device.acquire_next_image(self.swapchain, 0, self.available[frame], vk::Fence::null()) } {
            Ok((index, _)) => Ok(Some(index)),
            Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub(super) fn available(&self, frame: usize) -> vk::Semaphore {
        self.available[frame]
    }

    pub(super) fn finished(&self, frame: usize) -> vk::Semaphore {
        self.finished[frame]
    }

    /// where a finger at panel pixel (x, y) touches the bottom screen, in its 320x240, None outside of it
    pub(super) fn touch(&self, (x, y): (u32, u32), integer: bool) -> Option<(u16, u16)> {
        let (_, bottom) = layout(self.extent.width, self.extent.height, ScreenLayout::BottomOnly, self.orientation, super::TOP_WIDTH, integer);
        let bottom = bottom?;
        let (x, y) = (x as f32 - bottom.x, y as f32 - bottom.y);
        if x < 0.0 || y < 0.0 || x >= bottom.width || y >= bottom.height {
            return None;
        }
        let (u, v) = self.orientation.upright(x / bottom.width, y / bottom.height);
        Some((((u * 320.0) as u16).min(319), ((v * 240.0) as u16).min(239)))
    }

    /// draws the bottom screen from the set given, at the place of the screen in its image and its sample bounds, in
    /// the panel's image, or only clears it w/o a picture yet
    pub(super) fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer, image_index: u32, picture: Option<(vk::DescriptorSet, Vec<u8>)>, integer: bool) {
        let clear = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
        // safe bc the command buffer is recording and the panel's objects live as long as the panel
        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[image_index as usize])
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            let (_, bottom) = layout(self.extent.width, self.extent.height, ScreenLayout::BottomOnly, self.orientation, super::TOP_WIDTH, integer);
            if let (Some((set, constants)), Some(viewport)) = (picture, bottom) {
                set_viewport(device, command_buffer, viewport, self.extent);
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[set], &[]);
                device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::VERTEX, VERT_CONSTANTS_OFFSET, &self.orientation.clockwise_angle().to_le_bytes());
                device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::FRAGMENT, FRAG_CONSTANTS_OFFSET, &constants);
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
            }
            device.cmd_end_render_pass(command_buffer);
        }
    }

    pub(super) fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    /// destroys what the panel made and gives the display back to the lease, the device must be idle,
    /// handles still null get skipped by Vulkan
    pub(super) fn destroy(&mut self, device: &Device) {
        // safe bc the device is idle and each handle is either null or made by the panel
        unsafe {
            for semaphore in self.available.drain(..).chain(self.finished.drain(..)) {
                device.device.destroy_semaphore(semaphore, None);
            }
            for framebuffer in self.framebuffers.drain(..) {
                device.device.destroy_framebuffer(framebuffer, None);
            }
            for view in self.image_views.drain(..) {
                device.device.destroy_image_view(view, None);
            }
            device.swapchain_device.destroy_swapchain(self.swapchain, None);
            device.device.destroy_pipeline(self.pipeline, None);
            device.device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.device.destroy_render_pass(self.render_pass, None);
            device.surface_instance.destroy_surface(self.surface, None);
            if self.display != vk::DisplayKHR::null() {
                let released = (self.direct_mode.fp().release_display_ext)(device.physical, self.display);
                if released != vk::Result::SUCCESS {
                    log::warn!("vkReleaseDisplayEXT: {released}");
                }
            }
        }
        self.swapchain = vk::SwapchainKHR::null();
        self.pipeline = vk::Pipeline::null();
        self.pipeline_layout = vk::PipelineLayout::null();
        self.render_pass = vk::RenderPass::null();
        self.surface = vk::SurfaceKHR::null();
        self.display = vk::DisplayKHR::null();
    }
}

/// the instance extensions a leased panel's surface needs, all of them when the Vulkan library has them, none otherwise
pub(super) fn instance_extensions(entry: &ash::Entry) -> Vec<*const c_char> {
    let needed = [ash::khr::display::NAME, ash::ext::direct_mode_display::NAME, ash::ext::acquire_drm_display::NAME];
    // safe bc it's a query of the library w/ no layer named
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let all = needed.iter().all(|name| available.iter().any(|extension| extension.extension_name_as_c_str() == Ok(*name)));
    if all { needed.iter().map(|name| name.as_ptr()).collect() } else { Vec::new() }
}

/// the second display, wanted or not, its lease being asked for and the panel once it shows
#[derive(Default)]
pub(super) struct SecondDisplay {
    wanted: bool,
    /// the instance was made w/ instance_extensions
    possible: bool,
    /// the compositor was asked since the second display was last wanted, so one that offers none gets asked once
    asked: bool,
    pending: Option<mpsc::Receiver<Option<Output>>>,
    panel: Option<Panel>,
    /// the panel failed to take a picture, it goes b4 the next frame
    lost: bool,
}

impl SecondDisplay {
    pub(super) fn new(possible: bool) -> SecondDisplay {
        SecondDisplay { possible, ..SecondDisplay::default() }
    }

    pub(super) fn set_wanted(&mut self, wanted: bool) {
        if wanted && !self.wanted {
            self.asked = false;
            if !self.possible {
                log::warn!("the Vulkan driver lacks VK_KHR_display, VK_EXT_direct_mode_display or VK_EXT_acquire_drm_display, the screens stay in the window");
            }
        }
        self.wanted = wanted;
    }

    pub(super) fn panel(&self) -> Option<&Panel> {
        self.panel.as_ref()
    }

    pub(super) fn lose(&mut self) {
        self.lost = true;
    }

    /// once a frame, b4 it's drawn: asks for the lease when the second display's wanted, makes the panel when the
    /// lease comes, lets the panel go when it's no longer wanted, failed or the compositor took it back
    pub(super) fn update(&mut self, device: &Device) {
        if let Some(answer) = &self.pending {
            match answer.try_recv() {
                Ok(Some(output)) => {
                    self.pending = None;
                    if self.wanted {
                        match Panel::new(device, &output) {
                            Ok(panel) => self.panel = Some(panel),
                            Err(error) => {
                                log::error!("the second display could not be set up, {error}");
                                drm_lease::release();
                            }
                        }
                    } else {
                        drm_lease::release();
                    }
                }
                Ok(None) => {
                    self.pending = None;
                    log::warn!("no second display was leased, the screens stay in the window");
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.wanted && self.possible && !self.asked && self.panel.is_none() && self.pending.is_none() {
            self.asked = true;
            self.pending = Some(drm_lease::request());
        }
        let ended = self.panel.is_some() && !drm_lease::alive();
        if ended {
            log::warn!("the compositor took the second display back");
        }
        if self.panel.is_some() && (!self.wanted || self.lost || ended) {
            self.close(device);
        }
    }

    /// lets the panel go and gives the lease back
    pub(super) fn close(&mut self, device: &Device) {
        self.lost = false;
        let Some(mut panel) = self.panel.take() else { return };
        // safe bc it waits for the device's work to finish, some of which draws on the panel
        if let Err(error) = unsafe { device.device.device_wait_idle() } {
            log::error!("waiting for the device b4 letting the second display go: {error}");
        }
        panel.destroy(device);
        drm_lease::release();
        log::info!("the second display is let go");
    }
}
