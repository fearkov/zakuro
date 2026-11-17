//! one Vulkan device for the renderer and presenter when they'd pick the same GPU,
//! so the screens the renderer draws get shown straight from its images instead of copied out to the host and back
//! up

use ash::vk;

/// a device, the instance it came from and its one queue, gone once the last of the renderer and presenter lets go of it
/// both record and submit on the window's thread, so the queue needs no lock
pub struct SharedDevice {
    pub(crate) _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) family: u32,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    /// does the device do logic ops, and count pipeline statistics
    pub(crate) logic_ops: bool,
    pub(crate) statistics: bool,
    /// the blend state draws set as they go
    pub(crate) dynamic: crate::raster::hardware::Dynamic,
}

impl SharedDevice {
    /// a device on the GPU the renderer would pick, w/ no presenter, to time showing the screens straight from the GPU w/o a window
    pub fn new() -> Result<SharedDevice, String> {
        crate::raster::hardware::own_device()
    }
}

impl std::fmt::Debug for SharedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDevice").field("family", &self.family).finish_non_exhaustive()
    }
}

impl Drop for SharedDevice {
    fn drop(&mut self) {
        // safe bc whoever made something from the device destroyed it b4 letting go, the instance's surface went w/ the presenter
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
