//! wiring a presentation backend to a winit window.

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
use zakuro_gpu::{PresentError, Presenter, RendererKind, ScreenImage};

// exactly one backend exists per process, so the size of the largest variant
// is not worth an extra allocation to avoid.
#[allow(clippy::large_enum_variant)]
pub enum Backend {
    OpenGl {
        presenter: GlPresenter,
        context: PossiblyCurrentContext,
        surface: Surface<WindowSurface>,
    },
    Vulkan(VulkanPresenter),
    /// no window output, used when a backend could not start but the emulator
    /// should still run.
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

    pub fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
    ) -> Result<(), PresentError> {
        match self {
            Backend::OpenGl {
                presenter,
                context,
                surface,
            } => {
                presenter.present(top, bottom)?;
                surface
                    .swap_buffers(context)
                    .map_err(|e| PresentError::Backend(e.to_string()))
            }
            Backend::Vulkan(presenter) => presenter.present(top, bottom),
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
            // any config will do, prefer more color bits if offered.
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
    // ask for 3.3 core, which is what the presentation shaders need.
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(raw_handle));
    let not_current = unsafe { display.create_context(&config, &context_attributes) }
        .map_err(|e| format!("could not create an OpenGL 3.3 context: {e}"))?;

    let surface_attributes = window
        .build_surface_attributes(Default::default())
        .map_err(|e| e.to_string())?;
    let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
        .map_err(|e| format!("could not create a window surface: {e}"))?;

    let context = not_current
        .make_current(&surface)
        .map_err(|e| format!("could not make the context current: {e}"))?;

    // present on vblank so the frontend's own pacing has something to work
    // against, a failure here is not fatal.
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
