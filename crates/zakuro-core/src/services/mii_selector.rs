//! the Mii selector, the library applet a title starts to have a Mii picked,
//! for a save file or a racer. there is no list to pick from, so it picks
//! the console's own Mii right away, the one Mii every console has.

use crate::memory::config::MAC_ADDRESS;
use crate::services::cfg::CONSOLE_HASH;

/// the Mii selector's two applet ids, as a system applet and as a title
/// starts it.
pub const APPLET_IDS: [u32; 2] = [0x202, 0x402];

/// a Mii the way titles keep it, its data, two bytes of padding and a
/// checksum of everything before it, big endian.
const STORE_DATA_SIZE: usize = 0x60;
const CHECKSUM: usize = 0x5E;

// where the fields of the selector's answer sit
const GUEST_MII_INDEX: usize = 0x08;
const MII: usize = 0x0C;
/// a guest Mii's name follows the Mii, then the answer ends.
const RESULT_SIZE: usize = 0x84;

/// what the selector hands back, a Mii picked, not a guest, the console's.
pub fn result() -> Vec<u8> {
    let mut result = vec![0; RESULT_SIZE];
    result[GUEST_MII_INDEX..GUEST_MII_INDEX + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    result[MII..MII + STORE_DATA_SIZE].copy_from_slice(&console_mii());
    result
}

/// a normal Mii and not a special one, made the day the 3DS came out, in
/// two second steps since 2010.
const MII_ID: u32 = 0x9000_0000 | (421 * 86_400 / 2);

/// the console's own Mii, named after the console's user as cfg names it,
/// with the face a new male Mii starts with.
fn console_mii() -> [u8; STORE_DATA_SIZE] {
    let mut mii = [0u8; STORE_DATA_SIZE];
    // the format's version, then made on a 3DS
    mii[0x00] = 3;
    mii[0x03] = 3 << 4;
    // made on this console, a title's Mii library tells by comparing these
    // bytes with the hash cfg answers, as it sits in memory
    mii[0x04..0x0C].copy_from_slice(&CONSOLE_HASH.to_le_bytes());
    mii[0x0C..0x10].copy_from_slice(&MII_ID.to_be_bytes());
    // with the console's MAC address
    mii[0x10..0x16].copy_from_slice(&MAC_ADDRESS);
    // male, no birthday and red as the favorite color are all zero
    put_name(&mut mii[0x1A..0x2E], "Zakuro");
    // height and build
    mii[0x2E] = 64;
    mii[0x2F] = 64;
    // face shape and color, wrinkles and makeup are zero, then the hair and
    // its color
    mii[0x32] = 33;
    mii[0x33] = 1;
    // the eyes' type, color, scale, aspect, rotation, spacing and height
    let eyes = pack(&[(2, 6), (0, 3), (4, 4), (3, 3), (4, 5), (2, 4), (12, 5)]);
    mii[0x34..0x38].copy_from_slice(&eyes.to_le_bytes());
    // eyebrows the same, with a bit unused before the rotation
    let eyebrows = pack(&[(6, 5), (1, 3), (4, 4), (3, 3), (0, 1), (6, 5), (2, 4), (10, 5)]);
    mii[0x38..0x3C].copy_from_slice(&eyebrows.to_le_bytes());
    let halfwords = [
        // the nose's type, scale and height
        pack(&[(1, 5), (4, 4), (9, 5)]),
        // the mouth's type, color, scale and aspect
        pack(&[(23, 6), (0, 3), (4, 4), (3, 3)]),
        // the mouth's height, then no mustache
        pack(&[(13, 5), (0, 3)]),
        // no beard, then the mustache's scale and height
        pack(&[(0, 3), (0, 3), (4, 4), (10, 5)]),
        // no glasses, their color, scale and height
        pack(&[(0, 4), (0, 3), (4, 4), (10, 5)]),
        // no mole, its scale and where it would be
        pack(&[(0, 1), (4, 4), (2, 5), (20, 5)]),
    ];
    for (i, halfword) in halfwords.into_iter().enumerate() {
        mii[0x3C + i * 2..0x3E + i * 2].copy_from_slice(&(halfword as u16).to_le_bytes());
    }
    // the user made it
    put_name(&mut mii[0x48..0x5C], "Zakuro");
    let checksum = crc16(&mii[..CHECKSUM]);
    mii[CHECKSUM..].copy_from_slice(&checksum.to_be_bytes());
    mii
}

/// fields of the given widths, the first in the lowest bits.
fn pack(fields: &[(u32, u32)]) -> u32 {
    fields.iter().rev().fold(0, |packed, &(value, bits)| packed << bits | value)
}

/// a name of up to ten UTF-16 units.
fn put_name(out: &mut [u8], name: &str) {
    for (i, unit) in name.encode_utf16().take(out.len() / 2).enumerate() {
        out[i * 2..i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
    }
}

/// CRC-16/XMODEM, what Mii data is checked with.
fn crc16(bytes: &[u8]) -> u16 {
    bytes.iter().fold(0, |crc, &byte| {
        (0..8).fold(crc ^ (byte as u16) << 8, |crc, _| {
            if crc & 0x8000 != 0 { crc << 1 ^ 0x1021 } else { crc << 1 }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checksum_is_crc16_xmodem() {
        assert_eq!(crc16(b"123456789"), 0x31C3);
    }

    #[test]
    fn fields_pack_from_the_lowest_bits() {
        assert_eq!(pack(&[(1, 5), (4, 4), (9, 5)]), 1 | 4 << 5 | 9 << 9);
    }

    /// the answer reads as a Mii picked, with data a title's checks take.
    #[test]
    fn the_selector_answers_with_the_consoles_mii() {
        let result = result();
        assert_eq!(result.len(), RESULT_SIZE);
        assert_eq!(result[..8], [0; 8], "a Mii was picked, not a guest");
        let mii = &result[MII..MII + STORE_DATA_SIZE];
        assert_eq!(mii[0], 3);
        assert_eq!(u16::from_be_bytes([mii[CHECKSUM], mii[CHECKSUM + 1]]), crc16(&mii[..CHECKSUM]));
        let name: Vec<u16> = mii[0x1A..0x2E].as_chunks::<2>().0.iter().map(|&u| u16::from_le_bytes(u)).collect();
        assert_eq!(String::from_utf16_lossy(&name).trim_end_matches('\0'), "Zakuro");
        // the eyes' height, the last of their fields
        assert_eq!(u32::from_le_bytes(mii[0x34..0x38].try_into().unwrap()) >> 25 & 0x1F, 12);
        // made on this console, Super Mario 3D Land's Mii library compares
        // these eight bytes with the hash it got from cfg
        assert_eq!(mii[0x04..0x0C], CONSOLE_HASH.to_le_bytes());
    }
}
