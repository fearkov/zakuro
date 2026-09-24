//! the two read-only pages the kernel exposes to every process.

use zakuro_common::memory_map::{CONFIG_MEM_SIZE, SHARED_PAGE_SIZE};
use zakuro_common::ConsoleModel;

/// seconds between the 3DS epoch (1900-01-01) and the Unix epoch.
const EPOCH_OFFSET_SECONDS: u64 = 2_208_988_800;

fn write_u32(page: &mut [u8], offset: usize, value: u32) {
    page[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(page: &mut [u8], offset: usize, value: u64) {
    page[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// fills the configuration page.
pub fn init_config_mem(
    page: &mut [u8],
    model: ConsoleModel,
    app_mem_type: u32,
    app: u32,
    sys: u32,
    base: u32,
) {
    assert_eq!(page.len(), CONFIG_MEM_SIZE as usize);
    page.fill(0);

    // report firmware 11.x, which is what every retail console ended on.
    page[0x00] = 0x00; // kernel version revision
    page[0x01] = 0x34; // kernel version minor
    page[0x02] = 0x02; // kernel version major
    write_u32(page, 0x04, 0); // update flag
    write_u64(page, 0x08, 0); // NS title id
    write_u32(page, 0x10, 0x0000_0002); // syscore version
    page[0x14] = 0x01; // env info, production
    // unit info bit 0 distinguishes retail from a development unit.
    page[0x15] = 0x01;
    page[0x16] = 0x00; // previous firm
    write_u32(page, 0x18, 0x0000_F297); // CTR SDK version

    write_u32(page, 0x30, app_mem_type);
    write_u32(page, 0x40, app);
    write_u32(page, 0x44, sys);
    write_u32(page, 0x48, base);

    page[0x60] = 0x00;
    page[0x61] = 0x34;
    page[0x62] = 0x02;
    write_u32(page, 0x64, 0x0000_0002);
    write_u32(page, 0x68, 0x0000_F297);

    let _ = model;
}

/// live state.
pub fn init_shared_page(page: &mut [u8], model: ConsoleModel, slider_3d: f32) {
    assert_eq!(page.len(), SHARED_PAGE_SIZE as usize);
    page.fill(0);

    write_u32(page, 0x00, 0); // date/time selector, slot 0 is current
    page[0x04] = 1; // running hardware, retail product
    page[0x05] = if model.is_new3ds() { 2 } else { 1 };

    update_datetime(page, 0);

    // a plausible MAC. Games only ever show it or hash it.
    page[0x60..0x66].copy_from_slice(&[0x40, 0xF4, 0x07, 0x00, 0x00, 0x01]);
    page[0x66] = 3; // full wifi signal
    page[0x67] = 2; // wifi enabled and connected

    page[0x70..0x74].copy_from_slice(&slider_3d.to_le_bytes());
    page[0x74] = (slider_3d > 0.0) as u8; // 3D LED
    page[0x75] = 0x1F; // battery, charged, not charging
    page[0xB0] = 0; // no headset
}

/// refreshes the clock fields.
pub fn update_datetime(page: &mut [u8], tick: u64) {
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let date_time = unix + EPOCH_OFFSET_SECONDS * 1000;

    for slot in [0x20usize, 0x40] {
        write_u64(page, slot, date_time);
        write_u64(page, slot + 0x08, tick);
        // the ARM11 timer runs at 268.111856 MHz / 2.
        write_u64(page, slot + 0x10, 0x0000_0000_0001_0000);
        write_u64(page, slot + 0x18, 0);
    }
}

/// updates the 3D slider position, which games poll every frame.
pub fn set_slider_3d(page: &mut [u8], value: f32) {
    page[0x70..0x74].copy_from_slice(&value.clamp(0.0, 1.0).to_le_bytes());
    page[0x74] = (value > 0.0) as u8;
}
