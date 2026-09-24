//! cfg:u, system configuration blocks (language, region, user name).

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

/// region codes as cfg reports them.
pub const REGION_JAPAN: u8 = 0;
pub const REGION_USA: u8 = 1;
pub const REGION_EUROPE: u8 = 2;

/// language codes.
pub const LANGUAGE_ENGLISH: u8 = 1;

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    match header.command_id() {
        // GetConfigInfoBlk2(size, block id) with the result written to a
        // static buffer.
        0x0001 => {
            let size = buffer.get(&mut system.memory, 1);
            let block_id = buffer.get(&mut system.memory, 2);
            let out = buffer.get(&mut system.memory, 4);
            write_config_block(system, block_id, out, size);
            buffer.reply(&mut system.memory, 0x0001, &[]);
            true
        }
        // SecureInfoGetRegion
        0x0002 => {
            buffer.reply(&mut system.memory, 0x0002, &[system.config.region as u32]);
            true
        }
        // GenHashConsoleUnique(salt) -> a 64-bit hash.
        0x0003 => {
            buffer.reply(&mut system.memory, 0x0003, &[0x3341_5A55, 0x0000_5255]);
            true
        }
        // GetRegionCanadaUSA
        0x0004 => {
            let is_americas = system.config.region == REGION_USA;
            buffer.reply(&mut system.memory, 0x0004, &[is_americas as u32]);
            true
        }
        // GetSystemModel, 0 = Old3DS, 2 = Old2DS, 3 = New3DS.
        0x0005 => {
            let model = if system.config.new3ds { 3 } else { 0 };
            buffer.reply(&mut system.memory, 0x0005, &[model]);
            true
        }
        // GetModelNintendo2DS, 1 means "not a 2DS".
        0x0006 => {
            buffer.reply(&mut system.memory, 0x0006, &[1]);
            true
        }
        _ => false,
    }
}

/// factory values of the stereo camera configuration block.
const STEREO_CAMERA_SETTINGS: [f32; 8] = [
    62.0, 289.0, 76.8, 46.08, 10.0, 5.0, 55.58, 21.57,
];

/// writes one configuration block into guest memory.
fn write_config_block(system: &mut System, block_id: u32, out: u32, size: u32) {
    let mut data = vec![0u8; size as usize];
    match block_id {
        // user name, UTF-16, 0x1C bytes.
        0x000A_0000 => {
            let name: Vec<u16> = "Zakuro".encode_utf16().collect();
            for (i, unit) in name.iter().enumerate() {
                if i * 2 + 1 < data.len() {
                    data[i * 2..i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
                }
            }
        }
        // birthday, month then day.
        0x000A_0001 => {
            if data.len() >= 2 {
                data[0] = 1;
                data[1] = 1;
            }
        }
        // system language.
        0x000A_0002 => {
            if !data.is_empty() {
                data[0] = system.config.language;
            }
        }
        // country info, the last byte is the country code.
        0x000B_0000 | 0x000B_0001 => {
            if data.len() >= 4 {
                data[3] = if system.config.region == REGION_USA {
                    49 // united States
                } else {
                    110 // united Kingdom
                };
            }
        }
        // EULA version.
        0x000D_0000 => {
            if data.len() >= 2 {
                data[0] = 1;
                data[1] = 1;
            }
        }
        // parental controls, disabled.
        0x000C_0000 => {}
        // sound output mode, 1 = stereo.
        0x0007_0001 => {
            if !data.is_empty() {
                data[0] = 1;
            }
        }
        // stereoscopic 3D settings.
        0x0005_0001 => {}
        // the stereo camera's physical parameters, as eight floats, the
        // distance between the eyes and to the screen, the screen's size and so
        // on, in millimetres.
        0x0005_0005 => {
            // zeroes here turned the whole 3D camera into NaN. fuck this block
            for (chunk, value) in data.as_chunks_mut::<4>().0.iter_mut().zip(STEREO_CAMERA_SETTINGS) {
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        other => log::debug!("cfg: block 0x{other:08X} is not implemented, returning zeroes"),
    }
    system.memory.write_bytes(out, &data);
}
