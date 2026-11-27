//! wiring a presentation backend to a winit window

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use zakuro_gpu::backend::gl::GlPresenter;
use zakuro_gpu::backend::vulkan::VulkanPresenter;
use zakuro_gpu::{Overlay, PresentError, Presenter, RendererKind, ScreenImage, ScreenLayout, ScreenOrientation, SharedDevice};

// exactly one backend exists per process, so the size of the largest variant isn't worth an extra allocation to avoid
#[allow(clippy::large_enum_variant)]
pub enum Backend {
    OpenGl {
        presenter: GlPresenter,
        context: PossiblyCurrentContext,
        surface: Surface<WindowSurface>,
    },
    Vulkan(VulkanPresenter),
    /// no window output, used when a backend couldn't start but the emulator should still run
    None,
}

impl Backend {
    pub fn create(
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
        kind: RendererKind,
    ) -> Result<(Window, Backend), String> {
        match kind {
            RendererKind::OpenGl => create_opengl(event_loop, attributes),
            RendererKind::Vulkan => create_vulkan(event_loop, attributes),
            RendererKind::Software => {
                let window = event_loop
                    .create_window(attributes)
                    .map_err(|e| e.to_string())?;
                Ok((window, Backend::None))
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.name(),
            Backend::Vulkan(presenter) => presenter.name(),
            Backend::None => "none",
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        match self {
            Backend::OpenGl {
                presenter,
                context,
                surface,
            } => {
                presenter.resize(width, height);
                surface.resize(
                    context,
                    std::num::NonZeroU32::new(width).unwrap(),
                    std::num::NonZeroU32::new(height).unwrap(),
                );
            }
            Backend::Vulkan(presenter) => presenter.resize(width, height),
            Backend::None => {}
        }
    }

    /// does showing a frame wait for the display, the OpenGL swap waits for vblank
    pub fn waits_for_display(&self) -> bool {
        match self {
            Backend::OpenGl { .. } => true,
            Backend::Vulkan(presenter) => presenter.waits_for_display(),
            Backend::None => false,
        }
    }

    /// the Vulkan presenter's device, for the renderer to draw on so the screens get shown straight from it, when the two can share one
    pub fn shared_device(&self) -> Option<std::sync::Arc<SharedDevice>> {
        match self {
            Backend::Vulkan(presenter) => presenter.shared_device(),
            _ => None,
        }
    }

    pub fn shared_opengl_context(&self) -> Option<zakuro_gpu::SharedOpenGlContext> {
        match self {
            Backend::OpenGl { presenter, .. } => Some(presenter.shared_context()),
            _ => None,
        }
    }

    pub fn set_scaling(&mut self, filter: zakuro_gpu::ScreenFilter, integer: bool) {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.set_scaling(filter, integer),
            Backend::Vulkan(presenter) => presenter.set_scaling(filter, integer),
            Backend::None => {}
        }
    }

    pub fn set_stereo(&mut self, stereo: zakuro_gpu::Stereo) {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.set_stereo(stereo),
            Backend::Vulkan(presenter) => presenter.set_stereo(stereo),
            Backend::None => {}
        }
    }

    pub fn set_layout(&mut self, arrangement: ScreenLayout) {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.set_layout(arrangement),
            Backend::Vulkan(presenter) => presenter.set_layout(arrangement),
            Backend::None => {}
        }
    }

    pub fn set_orientation(&mut self, orientation: ScreenOrientation) {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.set_orientation(orientation),
            Backend::Vulkan(presenter) => presenter.set_orientation(orientation),
            Backend::None => {},
        }
    }

    pub fn set_top_width(&mut self, width: f32) {
        match self {
            Backend::OpenGl { presenter, .. } => presenter.set_top_width(width),
            Backend::Vulkan(presenter) => presenter.set_top_width(width),
            Backend::None => {}
        }
    }

    pub fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError> {
        match self {
            Backend::OpenGl {
                presenter,
                context,
                surface,
            } => {
                presenter.present(top, bottom, right, overlay)?;
                surface
                    .swap_buffers(context)
                    .map_err(|e| PresentError::Backend(e.to_string()))
            }
            Backend::Vulkan(presenter) => presenter.present(top, bottom, right, overlay),
            Backend::None => Ok(()),
        }
    }
}

fn create_opengl(
    event_loop: &ActiveEventLoop,
    attributes: WindowAttributes,
) -> Result<(Window, Backend), String> {
    let template = ConfigTemplateBuilder::new().with_alpha_size(8);
    let (window, config) = DisplayBuilder::new()
        .with_window_attributes(Some(attributes))
        .build(event_loop, template, |configs| {
            // any config will do, prefer more color bits if offered
            configs
                .reduce(|best, candidate| {
                    if candidate.num_samples() > best.num_samples() {
                        candidate
                    } else {
                        best
                    }
                })
                .expect("at least one GL config")
        })
        .map_err(|e| format!("no OpenGL configuration: {e}"))?;

    let window = window.ok_or("glutin did not create a window")?;
    let raw_handle = window
        .window_handle()
        .map_err(|e| e.to_string())?
        .as_raw();

    let display = config.display();
    // prefer 4.3 for the hardware rasterizer, while retaining 3.3 presentation on older drivers
    let make_attributes = |version| {
        ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(version)))
            .build(Some(raw_handle))
    };
    let not_current = match unsafe {
        display.create_context(&config, &make_attributes(Version::new(4, 3)))
    } {
        Ok(context) => context,
        Err(error_43) => {
            log::info!("OpenGL 4.3 unavailable ({error_43}), trying OpenGL 3.3 for presentation");
            unsafe { display.create_context(&config, &make_attributes(Version::new(3, 3))) }
                .map_err(|error| format!("could not create an OpenGL 4.3 or 3.3 context: {error}"))?
        }
    };

    let surface_attributes = window
        .build_surface_attributes(Default::default())
        .map_err(|e| e.to_string())?;
    let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
        .map_err(|e| format!("could not create a window surface: {e}"))?;

    let context = not_current
        .make_current(&surface)
        .map_err(|e| format!("could not make the context current: {e}"))?;

    // present on vblank so the frontend's own pacing has something to work against, a failure here isn't fatal
    let _ = surface.set_swap_interval(
        &context,
        SwapInterval::Wait(std::num::NonZeroU32::new(1).unwrap()),
    );

    let size = window.inner_size();
    let presenter = unsafe {
        GlPresenter::new(
            |name| {
                let name = std::ffi::CString::new(name).unwrap();
                display.get_proc_address(&name)
            },
            (size.width.max(1), size.height.max(1)),
        )
    }
    .map_err(|e| e.to_string())?;

    Ok((
        window,
        Backend::OpenGl {
            presenter,
            context,
            surface,
        },
    ))
}

fn create_vulkan(
    event_loop: &ActiveEventLoop,
    attributes: WindowAttributes,
) -> Result<(Window, Backend), String> {
    let window = event_loop
        .create_window(attributes)
        .map_err(|e| e.to_string())?;
    let size = window.inner_size();
    let presenter = VulkanPresenter::new(&window, (size.width.max(1), size.height.max(1)))
        .map_err(|e| e.to_string())?;
    Ok((window, Backend::Vulkan(presenter)))
}
