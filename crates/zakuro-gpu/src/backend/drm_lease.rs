//! the second panel's lease and its touchscreen, through drm_lease/ (Azahar's lease code, see the README there)

use std::ffi::{c_char, c_int, c_uint, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

/// a leased panel, as drm_lease.h's DrmLeaseOutput
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Output {
    /// the lease, open till release
    pub fd: c_int,
    pub connector_id: u32,
    /// the panel's mode, as it scans out
    pub width: u32,
    pub height: u32,
    /// degrees counter-clockwise
    pub rotation: c_int,
}

extern "C" {
    fn drm_lease_acquire(connector: *const c_char, rotation: c_int, touch_device: *const c_char, out: *mut Output) -> bool;
    fn drm_lease_alive() -> bool;
    fn drm_lease_release();
}

/// the finger on the panel, the top bit set while it's down, x in the bits above 32 and y in the ones below
static TOUCH: AtomicU64 = AtomicU64::new(0);
const DOWN: u64 = 1 << 63;

/// called by drm_lease_touch.cpp's thread for each touch report
#[no_mangle]
extern "C" fn second_display_touch(down: bool, x: c_uint, y: c_uint) {
    let packed = if down { DOWN | (u64::from(x) << 32) | u64::from(y) } else { 0 };
    TOUCH.store(packed, Ordering::Relaxed);
}

/// where a finger is on the panel, in pixels of its scanout
pub(super) fn touch() -> Option<(u32, u32)> {
    let packed = TOUCH.load(Ordering::Relaxed);
    (packed & DOWN != 0).then(|| (((packed & !DOWN) >> 32) as u32, packed as u32))
}

/// asks the compositor for the panel on a thread of its own, since that can take up to two 5 s timeouts,
/// the answer is None when nothing got leased, and the channel closes w/o one if the thread couldn't start
pub(super) fn request() -> mpsc::Receiver<Option<Output>> {
    let (answer, answered) = mpsc::channel();
    let started = std::thread::Builder::new()
        .name("drm lease".to_owned())
        .spawn(move || {
            // the receiver's gone only once the presenter is, on the way out
            let _ = answer.send(acquire());
        });
    if let Err(error) = started {
        log::error!("could not start the thread that leases the second display: {error}");
    }
    answered
}

/// the lease, SECOND_SCREEN_CONNECTOR names the connector (the first internal panel offered otherwise), SECOND_SCREEN_ROTATION the degrees the picture's turned
/// counter-clockwise (the connector's panel orientation otherwise), SECOND_SCREEN_TOUCH the touchscreen (auto,
/// a device path or name, or off)
fn acquire() -> Option<Output> {
    let setting = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let connector = setting("SECOND_SCREEN_CONNECTOR").and_then(|name| CString::new(name).ok());
    let rotation = setting("SECOND_SCREEN_ROTATION").and_then(|degrees| degrees.parse().ok()).unwrap_or(-1);
    let touch = setting("SECOND_SCREEN_TOUCH").unwrap_or_else(|| "auto".to_owned());
    let touch = (touch != "off").then(|| CString::new(touch).ok()).flatten();
    let pointer = |value: &Option<CString>| value.as_ref().map_or(std::ptr::null(), |value| value.as_ptr());
    let mut output = Output::default();
    // safe bc the strings outlive the call and output is a DrmLeaseOutput
    let leased = unsafe { drm_lease_acquire(pointer(&connector), rotation, pointer(&touch), &mut output) };
    leased.then_some(output)
}

/// is the lease still held, it ends when the compositor takes the panel back
pub(super) fn alive() -> bool {
    // safe bc it only reads the compositor's events, on the thread that releases the lease
    unsafe { drm_lease_alive() }
}

/// gives the panel back to the compositor and stops reading its touchscreen
pub(super) fn release() {
    // safe bc it's called once per lease got, after the surface on it is gone
    unsafe { drm_lease_release() };
    TOUCH.store(0, Ordering::Relaxed);
}
