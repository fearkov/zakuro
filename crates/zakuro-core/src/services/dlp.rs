//! dlp:SRVR, the host side of Download Play
//! a game offers its child (the lil program its card carries for others), takes guests in, sends them the child,
//! then the passphrase of the network it puts up for them
//! guests are Zakuro's own Download Play (see download_play) in the room local wireless goes thru
//! the card's child is encrypted, so guests only get it when the host has it decrypted (see `Config::dlp_child`),
//! otherwise a guest needs its own

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use zakuro_common::result::{Level, Module, ResultCode, Summary};

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::kernel::thread::nanos_to_ticks;
use crate::room::{Mac, BROADCAST};
use crate::services::uds::Wireless;
use crate::System;

// server states as GetServerState reports them
const NOT_INITIALIZED: u8 = 0;
const INITIALIZED: u8 = 1;
const ACCEPTING: u8 = 2;
const WAITING_FOR_DISTRIBUTION: u8 = 6;
const DISTRIBUTING: u8 = 7;
const NEED_TO_SEND_PASSPHRASE: u8 = 8;
const COMPLETE: u8 = 9;

/// where each guest's at
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Stage {
    /// waiting for the game to let it in
    NeedsAuth,
    Accepted,
    /// told distribution starts, waiting to hear if it needs the child
    NeedsDistributeAck,
    /// getting the child
    NeedsContent,
    /// has the child, waiting for the game to start
    DistributeDone,
    SentPassphrase,
}

impl Stage {
    /// what GetClientState reports for it
    fn reported(self) -> u32 {
        match self {
            Stage::NeedsAuth => 64,
            Stage::Accepted => 5,
            Stage::NeedsDistributeAck | Stage::NeedsContent => 6,
            Stage::DistributeDone => 7,
            Stage::SentPassphrase => 9,
        }
    }

    fn joined(self) -> bool {
        self != Stage::NeedsAuth
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Guest {
    node_id: u16,
    mac: Mac,
    node: Vec<u8>,
    stage: Stage,
    /// how much of the child it got so far
    sent: u64,
}

/// what a host offers, guests see it in their list
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    /// the child's title id
    pub child: u64,
    /// its SMDH (titles and icon), empty when the card has none
    pub smdh: Vec<u8>,
    pub size: u64,
    pub guests: u8,
    pub most: u8,
    /// can the host send the child (it has it decrypted)
    pub sends_child: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DlpState {
    state: u8,
    most: u8,
    manual: bool,
    guests: Vec<Guest>,
    /// the child the card carries
    child: Option<(u64, Vec<u8>, u64)>,
    next_offer: Option<u64>,
    /// the decrypted child, read the first time it's sent
    #[serde(skip)]
    decrypted: Option<Arc<Vec<u8>>>,
}

/// the child goes in pieces this big
pub const PIECE: usize = 32 * 1024;
/// and this many at a time to each guest, so a room keeps up
const PIECES_A_TICK: usize = 8;
/// an offer goes out once a second
const OFFER_NANOS: u64 = 1_000_000_000;

fn result(description: u32, summary: Summary, level: Level) -> u32 {
    ResultCode::new(level, summary, Module::Dlp, description).0
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    // w/o a room wireless is off, like local wireless does, unless the game already started on it in one
    if system.services.dlp.state == NOT_INITIALIZED && system.room.as_ref().is_none_or(|room| !room.open()) {
        return false;
    }
    let command = header.command_id();
    let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);
    match command {
        // Initialize(shared mem size, max guests, process, ...) and InitializeWithName(..., name), the child gets read from the card
        0x0001 | 0x000F => {
            let most = word(system, 2) as u8;
            initialize(system, most);
            buffer.reply(&mut system.memory, command, &[]);
        }
        // Finalize
        0x0002 => {
            tell_guests(system, &Wireless::DlpLeave);
            system.services.dlp = DlpState::default();
            buffer.reply(&mut system.memory, command, &[]);
        }
        // GetServerState
        0x0003 => {
            let state = system.services.dlp.state as u32;
            buffer.reply(&mut system.memory, command, &[state]);
        }
        // GetEventDescription, nothing to describe
        0x0004 => buffer.reply(&mut system.memory, command, &[0; 6]),
        // StartAccepting(manual, channel)
        0x0005 => {
            let (manual, channel) = (word(system, 1) & 0xFF != 0, word(system, 2) & 0xFF);
            if ![0, 1, 6, 11].contains(&channel) {
                buffer.reply_error(&mut system.memory, command, result(1021, Summary::InvalidArgument, Level::Permanent));
                return true;
            }
            let tick = system.cpu.cycles;
            let dlp = &mut system.services.dlp;
            dlp.manual = manual;
            dlp.state = ACCEPTING;
            dlp.next_offer = Some(tick);
            log::info!("dlp: offering the child to guests");
            buffer.reply(&mut system.memory, command, &[]);
        }
        // EndAccepting
        0x0006 => {
            let dlp = &mut system.services.dlp;
            dlp.state = INITIALIZED;
            dlp.next_offer = None;
            buffer.reply(&mut system.memory, command, &[]);
        }
        // StartDistribution
        0x0007 => {
            let dlp = &mut system.services.dlp;
            dlp.state = WAITING_FOR_DISTRIBUTION;
            dlp.next_offer = None;
            let mut told = Vec::new();
            for guest in dlp.guests.iter_mut().filter(|guest| guest.stage == Stage::Accepted) {
                guest.stage = Stage::NeedsDistributeAck;
                told.push(guest.mac);
            }
            for mac in told {
                send(system, mac, &Wireless::DlpStart);
            }
            settle(system);
            buffer.reply(&mut system.memory, command, &[]);
        }
        // SendWirelessRebootPassphrase(passphrase), the network the game puts
        // up next
        0x0008 => {
            let passphrase: Vec<u8> = (1..4).flat_map(|i| word(system, i).to_le_bytes()).take(9).collect();
            let guests = &system.services.dlp.guests;
            if guests.iter().any(|guest| guest.stage != Stage::DistributeDone) {
                buffer.reply_error(&mut system.memory, command, result(1002, Summary::InvalidState, Level::Status));
                return true;
            }
            let macs: Vec<Mac> = guests.iter().map(|guest| guest.mac).collect();
            for mac in macs {
                send(system, mac, &Wireless::DlpBegin { passphrase: passphrase.clone() });
            }
            let dlp = &mut system.services.dlp;
            for guest in &mut dlp.guests {
                guest.stage = Stage::SentPassphrase;
            }
            dlp.state = COMPLETE;
            log::info!("dlp: the guests have the passphrase, the game begins");
            buffer.reply(&mut system.memory, command, &[]);
        }
        // AcceptClient(node)
        0x0009 => {
            let node_id = word(system, 1) as u16;
            let found = system.services.dlp.guests.iter_mut().find(|guest| guest.node_id == node_id).map(|guest| {
                if guest.stage == Stage::NeedsAuth {
                    guest.stage = Stage::Accepted;
                }
                guest.mac
            });
            match found {
                Some(mac) => {
                    send(system, mac, &Wireless::DlpTaken { node_id });
                    buffer.reply(&mut system.memory, command, &[]);
                }
                None => buffer.reply_error(&mut system.memory, command, result(1018, Summary::NotFound, Level::Status)),
            }
        }
        // DisconnectClient(node)
        0x000A => {
            let node_id = word(system, 1) as u16;
            let dlp = &mut system.services.dlp;
            match dlp.guests.iter().position(|guest| guest.node_id == node_id) {
                Some(index) => {
                    let guest = dlp.guests.remove(index);
                    send(system, guest.mac, &Wireless::DlpLeave);
                    settle(system);
                    buffer.reply(&mut system.memory, command, &[]);
                }
                None => buffer.reply_error(&mut system.memory, command, result(1018, Summary::NotFound, Level::Status)),
            }
        }
        // GetConnectingClients(count, buffer), the guests' nodes
        0x000B => {
            let (room, pointer) = (word(system, 1) as usize, word(system, 3));
            let nodes: Vec<u16> = system.services.dlp.guests.iter().map(|guest| guest.node_id).collect();
            let bytes: Vec<u8> = nodes.iter().take(room).flat_map(|node| node.to_le_bytes()).collect();
            system.memory.write_bytes(pointer, &bytes);
            let descriptor = word(system, 2);
            buffer.set(&mut system.memory, 0, Header::new(command, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, nodes.len() as u32);
            buffer.set(&mut system.memory, 3, descriptor);
            buffer.set(&mut system.memory, 4, pointer);
        }
        // GetClientInfo(node), its name and its number
        0x000C => {
            let node_id = word(system, 1) as u16;
            match system.services.dlp.guests.iter().find(|guest| guest.node_id == node_id) {
                Some(guest) => {
                    let mut info = [0u8; 0x28];
                    info[0..8].copy_from_slice(guest.node.get(0..8).unwrap_or(&[0; 8]));
                    info[0x0C..0x20].copy_from_slice(guest.node.get(0x08..0x1C).unwrap_or(&[0; 0x14]));
                    info[0x24..0x28].copy_from_slice(&(node_id as u32).to_le_bytes());
                    let words: Vec<u32> = info.chunks(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect();
                    buffer.reply(&mut system.memory, command, &words);
                }
                None => buffer.reply_error(&mut system.memory, command, result(1007, Summary::NotFound, Level::Status)),
            }
        }
        // GetClientState(node), where it is and how much it has
        0x000D => {
            let node_id = word(system, 1) as u16;
            let dlp = &system.services.dlp;
            match dlp.guests.iter().find(|guest| guest.node_id == node_id) {
                Some(guest) => {
                    let state = guest.stage.reported() << 24 | (guest.stage.joined() as u32) << 16 | node_id as u32;
                    let total = dlp.child.as_ref().map_or(0, |(_, _, size)| size.div_ceil(PIECE as u64)) as u32;
                    let done = match guest.stage {
                        Stage::NeedsContent => guest.sent.div_ceil(PIECE as u64) as u32,
                        Stage::DistributeDone | Stage::SentPassphrase => total,
                        _ => 0,
                    };
                    buffer.reply(&mut system.memory, command, &[state, total, done, 0, 0]);
                }
                None => buffer.reply_error(&mut system.memory, command, result(1007, Summary::NotFound, Level::Status)),
            }
        }
        // IsChild(process), is the running title a Download Play child
        0x000E => {
            let child = system.title.as_ref().is_some_and(|title| title.program_id() >> 32 == 0x0004_0001);
            buffer.reply(&mut system.memory, command, &[child as u32]);
        }
        // GetDupNoticeNeed
        0x0010 => buffer.reply(&mut system.memory, command, &[0]),
        _ => return false,
    }
    true
}

/// starts the service and reads the child off the card
fn initialize(system: &mut System, most: u8) {
    let child = system.title.as_ref().and_then(|title| {
        let image = title.image().data();
        let (offset, size) = zakuro_fs::dlp::child_cia(image)?;
        let cia = image.get(offset as usize..(offset + size) as usize)?;
        let title_id = zakuro_fs::Cia::read(cia).ok()?.title_id;
        let smdh = zakuro_fs::dlp::cia_icon(cia).map(<[u8]>::to_vec).unwrap_or_default();
        Some((title_id, smdh, size))
    });
    match &child {
        Some((title_id, _, size)) => log::info!("dlp: the card's child is {title_id:016X}, {} KiB", size / 1024),
        None => log::warn!("dlp: the card carries no Download Play child"),
    }
    system.services.dlp = DlpState { state: INITIALIZED, most: most.clamp(1, 15), child, ..DlpState::default() };
}

fn send(system: &System, to: Mac, message: &Wireless) {
    let Some(room) = &system.room else { return };
    match postcard::to_allocvec(message) {
        Ok(body) => room.send(to, body),
        Err(error) => log::warn!("dlp: could not write a message, {error}"),
    }
}

fn tell_guests(system: &System, message: &Wireless) {
    for guest in &system.services.dlp.guests {
        send(system, guest.mac, message);
    }
}

/// once distribution started the server's state follows its guests
fn settle(system: &mut System) {
    let dlp = &mut system.services.dlp;
    if !matches!(dlp.state, WAITING_FOR_DISTRIBUTION | DISTRIBUTING) {
        return;
    }
    let pending = dlp.guests.iter().any(|guest| matches!(guest.stage, Stage::NeedsDistributeAck | Stage::NeedsContent));
    dlp.state = if !pending {
        NEED_TO_SEND_PASSPHRASE
    } else if dlp.guests.iter().any(|guest| guest.stage == Stage::NeedsContent) {
        DISTRIBUTING
    } else {
        WAITING_FOR_DISTRIBUTION
    };
}

/// the decrypted child, read once, none when there's nothing to send
fn decrypted(system: &mut System) -> Option<Arc<Vec<u8>>> {
    if let Some(bytes) = &system.services.dlp.decrypted {
        return Some(bytes.clone());
    }
    let path = system.config.dlp_child.clone()?;
    match std::fs::read(&path) {
        Ok(bytes) => {
            let bytes = Arc::new(bytes);
            system.services.dlp.decrypted = Some(bytes.clone());
            Some(bytes)
        }
        Err(error) => {
            log::warn!("dlp: could not read the decrypted child {}, {error}", path.display());
            None
        }
    }
}

/// what a guest sent the host
pub(crate) fn take(system: &mut System, from: Mac, message: Wireless) {
    let dlp = &system.services.dlp;
    if dlp.state == NOT_INITIALIZED {
        return;
    }
    match message {
        Wireless::DlpJoin { node } => {
            if let Some(guest) = dlp.guests.iter().find(|guest| guest.mac == from) {
                let node_id = guest.node_id;
                send(system, from, &Wireless::DlpTaken { node_id });
                return;
            }
            let free = (2..=dlp.most as u16 + 1).find(|id| dlp.guests.iter().all(|guest| guest.node_id != *id));
            let (Some(node_id), ACCEPTING) = (free, dlp.state) else {
                send(system, from, &Wireless::DlpRefused);
                return;
            };
            let stage = if dlp.manual { Stage::NeedsAuth } else { Stage::Accepted };
            system.services.dlp.guests.push(Guest { node_id, mac: from, node, stage, sent: 0 });
            log::info!("dlp: a guest joined as node {node_id}");
            if stage == Stage::Accepted {
                send(system, from, &Wireless::DlpTaken { node_id });
            }
        }
        Wireless::DlpNeed { needs } => {
            let has = needs && decrypted(system).is_some();
            let dlp = &mut system.services.dlp;
            let Some(guest) = dlp.guests.iter_mut().find(|guest| guest.mac == from && guest.stage == Stage::NeedsDistributeAck) else { return };
            if needs && !has {
                // nothing to send it, bye guest
                let mac = guest.mac;
                dlp.guests.retain(|guest| guest.mac != mac);
                send(system, mac, &Wireless::DlpRefused);
            } else {
                guest.stage = if needs { Stage::NeedsContent } else { Stage::DistributeDone };
                guest.sent = 0;
            }
            settle(system);
        }
        Wireless::DlpReady => {
            if let Some(guest) = system.services.dlp.guests.iter_mut().find(|guest| guest.mac == from && guest.stage == Stage::NeedsContent) {
                guest.stage = Stage::DistributeDone;
            }
            settle(system);
        }
        Wireless::DlpLeave => {
            system.services.dlp.guests.retain(|guest| guest.mac != from);
            settle(system);
        }
        _ => {}
    }
}

/// sends the offer when due and the next pieces of the child to guests getting it
pub fn tick(system: &mut System) {
    if system.room.is_none() || system.services.dlp.state == NOT_INITIALIZED {
        return;
    }
    let now = system.cpu.cycles;
    let dlp = &mut system.services.dlp;
    if let (Some(at), Some((child, smdh, size))) = (dlp.next_offer.filter(|&at| at <= now), &dlp.child) {
        let offer = Offer {
            child: *child,
            smdh: smdh.clone(),
            size: *size,
            guests: dlp.guests.len() as u8,
            most: dlp.most,
            sends_child: system.config.dlp_child.is_some(),
        };
        dlp.next_offer = Some(at.max(now) + nanos_to_ticks(OFFER_NANOS));
        send(system, BROADCAST, &Wireless::DlpOffer(offer));
    }
    if !system.services.dlp.guests.iter().any(|guest| guest.stage == Stage::NeedsContent) {
        return;
    }
    let Some(bytes) = decrypted(system) else { return };
    let mut pieces = Vec::new();
    for guest in system.services.dlp.guests.iter_mut().filter(|guest| guest.stage == Stage::NeedsContent) {
        for _ in 0..PIECES_A_TICK {
            if guest.sent >= bytes.len() as u64 {
                break;
            }
            let start = guest.sent as usize;
            let end = (start + PIECE).min(bytes.len());
            pieces.push((guest.mac, Wireless::DlpPiece { offset: start as u64, total: bytes.len() as u64, data: bytes[start..end].to_vec() }));
            guest.sent = end as u64;
        }
    }
    for (mac, piece) in pieces {
        send(system, mac, &piece);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download_play::{Guest as Visitor, Phase};
    use crate::room::Link;
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};

    fn request(system: &mut System, buffer: &CommandBuffer, words: &[u32]) -> (u32, Vec<u32>) {
        for (i, word) in words.iter().enumerate() {
            buffer.set(&mut system.memory, i as u32, *word);
        }
        handle_request(system, Target::service("dlp:SRVR".into(), 0));
        (buffer.get(&mut system.memory, 1), (2..7).map(|i| buffer.get(&mut system.memory, i)).collect())
    }

    /// a host offers its child, a guest joins, gets the child in pieces then the passphrase, and ends up w/ what the child runs w/
    #[test]
    fn a_guest_gets_the_child_and_the_passphrase() {
        let (host_link, guest_link) = Link::pair();
        let host_mac = host_link.mac();
        let (mut host, buffer) = system_with_thread();
        host.room = Some(host_link);
        let child: Vec<u8> = (0..100_000u32).map(|i| i as u8).collect();
        let path = std::env::temp_dir().join(format!("zakuro-dlp-child-{}.cia", std::process::id()));
        std::fs::write(&path, &child).unwrap();
        host.config.dlp_child = Some(path.clone());

        // Initialize(size, four guests, process, ...), the child like a card has it
        assert_eq!(request(&mut host, &buffer, &[0x0001_0183, 0x1000, 4, 0, 0, 0, 0, 0, 0, 0]).0, 0);
        host.services.dlp.child = Some((0x0004_0001_0010_BF00, vec![0x53; 0x36C0], child.len() as u64));
        assert_eq!(request(&mut host, &buffer, &[0x0003_0000]).1[0], INITIALIZED as u32);
        // StartAccepting(automatic, channel 1)
        assert_eq!(request(&mut host, &buffer, &[0x0005_0080, 0, 1]).0, 0);

        let mut guest = Visitor::new(guest_link);
        let settle = |host: &mut System, guest: &mut Visitor| {
            for _ in 0..40 {
                std::thread::sleep(std::time::Duration::from_millis(3));
                crate::services::uds::tick(host);
                tick(host);
                host.cpu.cycles += crate::CYCLES_PER_AUDIO_FRAME;
                guest.poll();
            }
        };
        settle(&mut host, &mut guest);
        assert_eq!(guest.offers().len(), 1);
        let heard = guest.offers()[0].clone();
        assert_eq!((heard.host, heard.offer.child, heard.offer.sends_child), (host_mac, 0x0004_0001_0010_BF00, true));

        guest.join(host_mac, crate::download_play::node(7, "Guest"), false);
        settle(&mut host, &mut guest);
        assert_eq!(guest.phase(), Some(&Phase::Waiting));
        let (code, words) = request(&mut host, &buffer, &[0x000D_0040, 2]);
        assert_eq!((code, words[0]), (0, 5 << 24 | 1 << 16 | 2), "node 2, joined");
        // GetClientInfo(node 2), the guest's number and name
        let (code, words) = request(&mut host, &buffer, &[0x000C_0040, 2]);
        assert_eq!((code, words[0], words[3]), (0, 7, u32::from_le_bytes(*b"G\0u\0")));

        // StartDistribution, the child goes in pieces
        assert_eq!(request(&mut host, &buffer, &[0x0007_0000]).0, 0);
        settle(&mut host, &mut guest);
        settle(&mut host, &mut guest);
        assert_eq!(guest.phase(), Some(&Phase::Ready));
        assert_eq!(request(&mut host, &buffer, &[0x0003_0000]).1[0], NEED_TO_SEND_PASSPHRASE as u32);

        // SendWirelessRebootPassphrase("abcdef12", padded)
        let passphrase = *b"abcdef12\0\0\0\0";
        let words: Vec<u32> = passphrase.chunks(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect();
        assert_eq!(request(&mut host, &buffer, &[0x0008_00C0, words[0], words[1], words[2]]).0, 0);
        settle(&mut host, &mut guest);
        match guest.phase() {
            Some(Phase::Begun { child: got, reboot }) => {
                assert_eq!(got.as_ref(), Some(&child));
                assert_eq!(&reboot[..6], &host_mac);
                assert_eq!(&reboot[6..15], b"abcdef12\0");
            }
            other => panic!("the guest is at {other:?}"),
        }
        assert_eq!(request(&mut host, &buffer, &[0x0003_0000]).1[0], COMPLETE as u32);
        let _ = std::fs::remove_file(path);
    }

    /// w/o a room wireless is off, like local wireless does
    #[test]
    fn no_room_is_wireless_off() {
        let (mut system, buffer) = system_with_thread();
        let (code, _) = request(&mut system, &buffer, &[0x0001_0183, 0x1000, 4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(code, zakuro_common::result::errors::UDS_WIRELESS_OFF.0);
    }
}
