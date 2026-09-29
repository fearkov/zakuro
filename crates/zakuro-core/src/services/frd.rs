//! frd:u, the friend list and this console's own friend details. the
//! console keeps them itself and answers offline too, as one that never
//! logged in and has no friends. what needs the servers is left to the
//! offline reply.

use zakuro_common::result::errors;

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::services::cfg::REGION_USA;
use crate::System;

/// the platform a profile names, the 3DS.
const PLATFORM_CTR: u32 = 2;

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let (region, language) = (system.config.region, system.config.language);
    let memory = &mut system.memory;
    match command {
        // HasLoggedIn and IsOnline, neither
        0x0001 | 0x0002 => buffer.reply(memory, command, &[0]),
        // Logout, AttachToEventNotification, SetNotificationMask and
        // SetClientSdkVersion
        0x0004 | 0x0020 | 0x0021 | 0x0032 => buffer.reply(memory, command, &[]),
        // GetMyFriendKey, a principal id, padding and a friend code seed,
        // none of which a console that never logged in has
        0x0005 => buffer.reply(memory, command, &[0; 4]),
        // GetMyPreference, private, showing neither game
        0x0006 => buffer.reply(memory, command, &[0; 3]),
        // GetMyProfile, region, country, area and language, then platform
        0x0007 => {
            let country: u32 = if region == REGION_USA { 49 } else { 110 };
            let first = region as u32 | country << 8 | (language as u32) << 24;
            buffer.reply(memory, command, &[first, PLATFORM_CTR]);
        }
        // GetMyPresence, into a static buffer that stays as it is
        0x0008 => buffer.reply(memory, command, &[]),
        // GetMyScreenName, the user's name in eleven UTF-16 units
        0x0009 => {
            let mut name = [0u8; 24];
            for (i, unit) in "Zakuro".encode_utf16().enumerate() {
                name[i * 2..i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
            }
            let words: Vec<u32> = name.as_chunks::<4>().0.iter().map(|&w| u32::from_le_bytes(w)).collect();
            buffer.reply(memory, command, &words);
        }
        // GetMyMii
        0x000A => buffer.reply(memory, command, &[0; 24]),
        // GetMyLocalAccountId
        0x000B => buffer.reply(memory, command, &[1]),
        // GetMyPlayingGame and GetMyFavoriteGame, a title id, its version
        // and a word more
        0x000C | 0x000D => buffer.reply(memory, command, &[0; 4]),
        // GetMyNcPrincipalId
        0x000E => buffer.reply(memory, command, &[0]),
        // GetMyComment
        0x000F => buffer.reply(memory, command, &[0; 9]),
        // GetFriendKeyList and GetEventNotification, none
        0x0011 | 0x0022 => buffer.reply(memory, command, &[0]),
        // lookups on friends the title names, of which it has none to name
        0x0012..=0x001A => buffer.reply(memory, command, &[]),
        // IsIncludedInFriendList
        0x001B => buffer.reply(memory, command, &[0]),
        // GetLastResponseResult, what the last request to the servers got
        0x0023 => buffer.reply(memory, command, &[errors::NOT_CONNECTED.0]),
        // ResultToErrorCode, no code to show
        0x0027 => buffer.reply(memory, command, &[0]),
        _ => return false,
    }
    true
}
