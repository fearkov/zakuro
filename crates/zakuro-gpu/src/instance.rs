//! the system's Vulkan library and an instance of it, on any host, macOS's
//! included, where Vulkan is MoltenVK over Metal and only shows up to an
//! instance that asks for portable drivers.

use ash::vk;
use std::ffi::{c_char, CStr};

/// where macOS keeps a Vulkan library when the loader's own name finds
/// none: Homebrew's on Apple silicon and on Intel, then MoltenVK by itself,
/// which a program can start straight from without the loader.
#[cfg(target_os = "macos")]
const MACOS_LIBRARIES: [&str; 4] = [
    "/opt/homebrew/lib/libvulkan.1.dylib",
    "/usr/local/lib/libvulkan.1.dylib",
    "/opt/homebrew/lib/libMoltenVK.dylib",
    "/usr/local/lib/libMoltenVK.dylib",
];

/// the MoltenVK Zakuro.app comes with, in its Frameworks folder, which asks
/// nothing of the computer.
#[cfg(target_os = "macos")]
fn bundled() -> Option<std::path::PathBuf> {
    let program = std::env::current_exe().ok()?;
    Some(program.parent()?.parent()?.join("Frameworks/libMoltenVK.dylib"))
}

/// the system's Vulkan library, by the loader's name, or on macOS the one
/// Zakuro.app comes with first, and after it from where Homebrew and the
/// Vulkan SDK put it. the error is the one for the loader's name, which says
/// where it looked.
pub(crate) fn entry() -> Result<ash::Entry, ash::LoadingError> {
    #[cfg(target_os = "macos")]
    if let Some(path) = bundled().filter(|path| path.is_file()) {
        // SAFETY: a Vulkan library at a path of its own
        if let Ok(entry) = unsafe { ash::Entry::load_from(path) } {
            return Ok(entry);
        }
    }
    // SAFETY: loads the system's Vulkan library, which is the documented way
    // to start
    let error = match unsafe { ash::Entry::load() } {
        Ok(entry) => return Ok(entry),
        Err(error) => error,
    };
    #[cfg(target_os = "macos")]
    for path in MACOS_LIBRARIES {
        // SAFETY: as above, a Vulkan library at a path of its own
        if let Ok(entry) = unsafe { ash::Entry::load_from(path) } {
            return Ok(entry);
        }
    }
    Err(error)
}

/// an instance with the extensions given, and the drivers that implement
/// Vulkan on top of another API, MoltenVK's, when the library lists them
/// apart, as the loader does from 1.3.216 on.
pub(crate) fn new(entry: &ash::Entry, application: &vk::ApplicationInfo, extensions: &[*const c_char]) -> ash::prelude::VkResult<ash::Instance> {
    // SAFETY: a query of the library, with no layer named
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let portable = available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME));
    let mut extensions = extensions.to_vec();
    let mut flags = vk::InstanceCreateFlags::empty();
    if portable {
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }
    let info = vk::InstanceCreateInfo::default().application_info(application).enabled_extension_names(&extensions).flags(flags);
    // SAFETY: extensions the library lists, the caller's included
    unsafe { entry.create_instance(&info, None) }
}

/// the extension a device made on a GPU has to be made with when the GPU
/// does only part of Vulkan, as MoltenVK's do, none on any other.
pub(crate) fn portability_subset(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<&'static CStr> {
    // SAFETY: a query on a GPU of the instance
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    available
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_subset::NAME))
        .then_some(ash::khr::portability_subset::NAME)
}
