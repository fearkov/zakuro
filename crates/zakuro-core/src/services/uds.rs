//! nwm::UDS, local wireless
//! consoles find the networks others put up, join them and pass data, here thru a room instead of the air (see room)
//! what goes between consoles is Zakuro's own, none of the 802.11 frames or their encryption, except the beacons a title reads itself
//! w/o a room wireless is off, like when a player switches it off

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use zakuro_common::result::{Level, Module, ResultCode, Summary};

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::kernel::thread::{nanos_to_ticks, ThreadId};
use crate::room::{Mac, BROADCAST};
use crate::System;

// what GetConnectionStatus says the console is
const NOT_CONNECTED: u32 = 3;
const HOST: u32 = 6;
const CONNECTING: u32 = 7;
const CLIENT: u32 = 9;
const SPECTATOR: u32 = 10;
// and why it last changed
const ESTABLISHED: u32 = 1;
const LOST: u32 = 4;

const NETWORK_SIZE: usize = 0x108;
const NODE_SIZE: usize = 0x28;
const MAX_NODES: usize = 16;
const BROADCAST_NODE: u16 = 0xFFFF;
const HOST_NODE: u16 = 1;
const DEFAULT_CHANNEL: u8 = 11;
const APPLICATION_DATA: usize = 0xC8;
/// a host's beacon goes out every 100 time units of 1.024 ms
const BEACON_NANOS: u64 = 102_400_000;
/// a scan takes about this long on the console
const SCAN_NANOS: u64 = 300_000_000;
/// joining waits this long for the host
const JOIN_NANOS: u64 = 5_000_000_000;
const MAX_BINDS: usize = 16;
const MIN_RECEIVE_BUFFER: u32 = 0x5F4;
const MAX_SEND: u32 = 0x5C6;
/// beacons kept between scans, one per host
const MAX_BEACONS: usize = 15;
const NINTENDO_OUI: [u8; 3] = [0x00, 0x1F, 0x32];
/// the encrypted data of a beacon gets split between two tags past this
const FIRST_TAG_DATA: usize = 0xFA;

// the results the console gives
const TOO_LARGE: u32 = 1001;
const NOT_AUTHORIZED: u32 = 1002;
const OUT_OF_MEMORY: u32 = 1011;
const NOT_FOUND: u32 = 1018;
const TIMEOUT: u32 = 1022;
const WRONG_STATUS: u32 = 490;

fn result(description: u32, summary: Summary, level: Level) -> u32 {
    ResultCode::new(level, summary, Module::Uds, description).0
}

/// what one console sends another thru the room
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Wireless {
    /// a host's beacon, as a title reads it from a scan
    Beacon { channel: u8, frame: Vec<u8> },
    /// a console asks to join the network a host has up
    Join { network_id: u32, spectator: bool, passphrase: Vec<u8>, node: Vec<u8> },
    /// the host lets it in, as a node
    Welcome { node_id: u16, network: Vec<u8>, peers: Vec<Peer> },
    /// or turns it away
    Refused,
    /// who's in the network now, whenever that changes
    Peers(Vec<Peer>),
    /// a client leaves, or the host ends the network or kicks a client
    Leave,
    /// data for a node of a network, or for all of them
    Data { network_id: u32, channel: u8, from: u16, to: u16, payload: Vec<u8> },
    /// a Download Play host's offer, every second while it takes guests
    DlpOffer(crate::services::dlp::Offer),
    /// a guest asks to take part, w/ its node
    DlpJoin { node: Vec<u8> },
    /// the host lets it in, as a node
    DlpTaken { node_id: u16 },
    /// or turns it away, or kicks it
    DlpRefused,
    /// the host starts sending the child
    DlpStart,
    /// does the guest need the child sent
    DlpNeed { needs: bool },
    /// a piece of the child
    DlpPiece { offset: u64, total: u64, data: Vec<u8> },
    /// the guest has the whole child
    DlpReady,
    /// the game starts, w/ the passphrase of the network the host puts up for the children
    DlpBegin { passphrase: Vec<u8> },
    /// either side leaves the session
    DlpLeave,
}

impl Wireless {
    /// is it Download Play's, the dlp service takes those
    fn download_play(&self) -> bool {
        matches!(
            self,
            Wireless::DlpOffer(_)
                | Wireless::DlpJoin { .. }
                | Wireless::DlpTaken { .. }
                | Wireless::DlpRefused
                | Wireless::DlpStart
                | Wireless::DlpNeed { .. }
                | Wireless::DlpPiece { .. }
                | Wireless::DlpReady
                | Wireless::DlpBegin { .. }
                | Wireless::DlpLeave
        )
    }
}

/// a node of a network, its addr and how its title describes it
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Peer {
    node_id: u16,
    mac: Mac,
    node: Vec<u8>,
}

/// what GetConnectionStatus answers
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Status {
    status: u32,
    reason: u32,
    node_id: u16,
    changed: u16,
    nodes: [u16; MAX_NODES],
    total: u8,
    max: u8,
    bitmask: u16,
}

impl Status {
    fn words(&self) -> [u32; 12] {
        let mut bytes = [0u8; 0x30];
        bytes[0..4].copy_from_slice(&self.status.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.reason.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.node_id.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.changed.to_le_bytes());
        for (i, node) in self.nodes.iter().enumerate() {
            bytes[12 + i * 2..14 + i * 2].copy_from_slice(&node.to_le_bytes());
        }
        bytes[0x2C] = self.total;
        bytes[0x2D] = self.max;
        bytes[0x2E..0x30].copy_from_slice(&self.bitmask.to_le_bytes());
        std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
    }

    /// the nodes of a network as the status lists them, what changed told apart from what it had
    fn take_peers(&mut self, peers: &[Peer]) {
        let before = self.bitmask;
        self.nodes = [0; MAX_NODES];
        self.bitmask = 0;
        for peer in peers.iter().filter(|peer| (1..=MAX_NODES as u16).contains(&peer.node_id)) {
            let index = peer.node_id as usize - 1;
            self.nodes[index] = peer.node_id;
            self.bitmask |= 1 << index;
        }
        self.total = self.bitmask.count_ones() as u8;
        self.changed |= before ^ self.bitmask;
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Bind {
    id: u32,
    channel: u8,
    /// the node it takes data from, any for BROADCAST_NODE
    from: u16,
    event: ObjectId,
    /// what came, from which node, oldest first
    queue: VecDeque<(u16, Vec<u8>)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Beacon {
    mac: Mac,
    channel: u8,
    frame: Vec<u8>,
}

/// a title waiting on ConnectNetwork2 for the host to answer
#[derive(Debug, Serialize, Deserialize)]
struct Joining {
    thread: ThreadId,
    event: ObjectId,
    until: u64,
    spectator: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct UdsState {
    initialized: bool,
    status_event: Option<ObjectId>,
    /// the console's own node, as its title describes it
    node: Vec<u8>,
    status: Status,
    /// the network hosted or joined, the host's addr, and the passphrase a host takes
    network: Vec<u8>,
    host: Option<Mac>,
    passphrase: Vec<u8>,
    /// the nodes in it, the host first
    peers: Vec<Peer>,
    binds: Vec<Bind>,
    beacons: VecDeque<Beacon>,
    next_beacon: Option<u64>,
    joining: Option<Joining>,
}

impl UdsState {
    fn network_id(&self) -> u32 {
        self.network.get(0x18..0x1C).map_or(0, |id| u32::from_be_bytes(id.try_into().unwrap()))
    }

    fn connected(&self) -> bool {
        matches!(self.status.status, HOST | CLIENT | SPECTATOR)
    }
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    // w/o a room wireless is off, like the offline answer has it, unless a title already started on it in one
    if !system.services.uds.initialized && system.room.as_ref().is_none_or(|room| !room.open()) {
        return false;
    }
    let command = header.command_id();
    let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);
    match command {
        // Initialize(size, node, block) / InitializeWithVersion(size, node, version, block)
        0x0001 | 0x001B => {
            let node: Vec<u8> = (2..12).flat_map(|i| word(system, i).to_le_bytes()).collect();
            let event = initialize(system, node);
            let handle = system.kernel.handles.create(&mut system.kernel.objects, event, "UDS status event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        // Scrap / Finalize
        0x0002 | 0x0003 => {
            leave(system);
            let uds = &mut system.services.uds;
            uds.initialized = false;
            uds.binds.clear();
            uds.beacons.clear();
            uds.status = Status { status: NOT_CONNECTED, ..Status::default() };
            buffer.reply(&mut system.memory, command, &[]);
        }
        // CreateNetwork2(passphrase size, network, passphrase)
        0x001D => {
            let pointer = word(system, 3);
            let network = read(system, pointer, NETWORK_SIZE);
            let size = word(system, 1).min(0xFF) as usize;
            let pointer = word(system, 5);
            let passphrase = read(system, pointer, size);
            let code = host(system, network, passphrase);
            reply_result(system, buffer, command, code);
        }
        // DestroyNetwork
        0x0008 => {
            if system.services.uds.status.status != HOST {
                reply_result(system, buffer, command, result(WRONG_STATUS, Summary::InvalidState, Level::Status));
                return true;
            }
            leave(system);
            reply_result(system, buffer, command, 0);
        }
        // DisconnectNetwork, a host may not
        0x000A => {
            if system.services.uds.status.status == HOST {
                reply_result(system, buffer, command, result(WRONG_STATUS, Summary::InvalidState, Level::Status));
                return true;
            }
            leave(system);
            reply_result(system, buffer, command, 0);
        }
        // EjectClient(node)
        0x0005 => {
            let node_id = word(system, 1) as u16;
            if system.services.uds.status.status == HOST {
                if let Some(peer) = system.services.uds.peers.iter().find(|peer| peer.node_id == node_id && node_id != HOST_NODE).cloned() {
                    send(system, peer.mac, &Wireless::Leave);
                    drop_peer(system, peer.mac);
                }
            }
            reply_result(system, buffer, command, 0);
        }
        // UpdateNetworkAttribute(bits, set)
        0x0007 => {
            let (bits, set) = (word(system, 1) as u16, word(system, 2) & 0xFF != 0);
            let uds = &mut system.services.uds;
            if uds.status.status == HOST && uds.network.len() == NETWORK_SIZE {
                let mut attributes = u16::from_be_bytes([uds.network[0x16], uds.network[0x17]]);
                if set {
                    attributes |= bits;
                } else {
                    attributes &= !bits;
                }
                uds.network[0x16..0x18].copy_from_slice(&attributes.to_be_bytes());
            }
            reply_result(system, buffer, command, 0);
        }
        // GetConnectionStatus, the changes it reports get told only once
        0x000B => {
            let status = system.services.uds.status;
            system.services.uds.status.changed = 0;
            buffer.reply(&mut system.memory, command, &status.words());
        }
        // two values no title acts on
        0x000C => buffer.reply(&mut system.memory, command, &[0, 0]),
        // GetNodeInformation(node)
        0x000D => {
            let node_id = word(system, 1) as u16;
            match system.services.uds.peers.iter().find(|peer| peer.node_id == node_id) {
                Some(peer) => {
                    let node = node_bytes(peer);
                    let words: Vec<u32> = node.chunks(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect();
                    buffer.reply(&mut system.memory, command, &words);
                }
                None => reply_result(system, buffer, command, result(NOT_FOUND, Summary::WrongArgument, Level::Usage)),
            }
        }
        // StartScan(size, scan input, comm id, id8, event, buffer)
        0x000F => {
            let size = word(system, 1);
            let mac: Vec<u8> = [4, 5].iter().flat_map(|&i| word(system, i).to_le_bytes()).take(6).collect();
            let pointer = word(system, 20);
            scan(system, pointer, size, mac.try_into().unwrap());
            reply_result(system, buffer, command, 0);
            // the radio listens a while b4 it answers
            let tick = system.cpu.cycles;
            system.kernel.sleep_current(nanos_to_ticks(SCAN_NANOS), tick);
        }
        // SetApplicationData(size, data)
        0x0010 => {
            let size = word(system, 1) as usize;
            if size > APPLICATION_DATA {
                reply_result(system, buffer, command, result(TOO_LARGE, Summary::WrongArgument, Level::Usage));
                return true;
            }
            let pointer = word(system, 3);
            let data = read(system, pointer, size);
            let uds = &mut system.services.uds;
            if uds.network.len() == NETWORK_SIZE {
                uds.network[0x3F] = size as u8;
                uds.network[0x40..0x40 + size].copy_from_slice(&data);
            }
            reply_result(system, buffer, command, 0);
        }
        // GetApplicationData(size), into the title's static buffer
        0x0011 => {
            let room = word(system, 1) as usize;
            let uds = &system.services.uds;
            let size = uds.network.get(0x3F).map_or(0, |&size| size as usize).min(APPLICATION_DATA);
            if room < size || size == 0 {
                buffer.reply(&mut system.memory, command, &[0]);
                return true;
            }
            let data = uds.network[0x40..0x40 + size].to_vec();
            let destination = reply_static(system, buffer, command, &[size as u32], &data, data.len());
            log::debug!("uds: the application data went to 0x{destination:08X}");
        }
        // Bind(bind id, receive buffer size, data channel, node)
        0x0012 => {
            let (id, size, channel, from) = (word(system, 1), word(system, 2), word(system, 3) as u8, word(system, 4) as u16);
            let code = if id == 0 || channel == 0 {
                Some(result(NOT_AUTHORIZED, Summary::WrongArgument, Level::Usage))
            } else if system.services.uds.binds.len() >= MAX_BINDS {
                Some(result(OUT_OF_MEMORY, Summary::OutOfResource, Level::Status))
            } else if size < MIN_RECEIVE_BUFFER {
                Some(result(TOO_LARGE, Summary::WrongArgument, Level::Usage))
            } else {
                None
            };
            if let Some(code) = code {
                reply_result(system, buffer, command, code);
                return true;
            }
            let event = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "UDS bind")));
            // the service keeps it for the bind's whole life
            system.kernel.objects.add_ref(event);
            system.services.uds.binds.retain(|bind| bind.id != id);
            system.services.uds.binds.push(Bind { id, channel, from, event, queue: VecDeque::new() });
            let handle = system.kernel.handles.create(&mut system.kernel.objects, event, "UDS bind event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        // Unbind(bind id)
        0x0013 => {
            let id = word(system, 1);
            if let Some(index) = system.services.uds.binds.iter().position(|bind| bind.id == id) {
                let bind = system.services.uds.binds.remove(index);
                system.kernel.objects.release(bind.event);
            }
            reply_result(system, buffer, command, 0);
        }
        // PullPacket(bind id, buffer words, buffer size)
        0x0014 => {
            let (id, words, most) = (word(system, 1), word(system, 2), word(system, 3));
            let room = (words.min(0x172) << 2) as usize;
            let uds = &mut system.services.uds;
            let Some(bind) = uds.binds.iter_mut().find(|bind| bind.id == id).filter(|_| matches!(uds.status.status, HOST | CLIENT | SPECTATOR)) else {
                reply_result(system, buffer, command, result(NOT_AUTHORIZED, Summary::InvalidState, Level::Status));
                return true;
            };
            match bind.queue.front() {
                Some((_, payload)) if payload.len() > most as usize => {
                    reply_result(system, buffer, command, result(TOO_LARGE, Summary::WrongArgument, Level::Usage));
                }
                Some(_) => {
                    let (from, payload) = bind.queue.pop_front().unwrap();
                    reply_static(system, buffer, command, &[payload.len() as u32, from as u32], &payload, room);
                }
                None => {
                    reply_static(system, buffer, command, &[0, 0], &[], room);
                }
            }
        }
        // SendTo(?, node, data channel, buffer words, size, flags, data)
        0x0017 => {
            let (to, channel, size, flags) = (word(system, 2) as u16, word(system, 3) as u8, word(system, 5), word(system, 6) as u8);
            let pointer = word(system, 8);
            let code = send_to(system, to, channel, size, flags, pointer);
            reply_result(system, buffer, command, code);
        }
        // GetChannel
        0x001A => {
            let uds = &system.services.uds;
            let channel = if uds.connected() { uds.network.get(6).copied().filter(|&c| c != 0).unwrap_or(DEFAULT_CHANNEL) } else { 0 };
            buffer.reply(&mut system.memory, command, &[channel as u32]);
        }
        // ConnectNetwork2(type, passphrase size, network, passphrase), returns once the host answers
        0x001E => {
            let spectator = word(system, 1) & 0xFF == 2;
            let size = word(system, 2).min(0xFF) as usize;
            let pointer = word(system, 4);
            let network = read(system, pointer, NETWORK_SIZE);
            let pointer = word(system, 6);
            let passphrase = read(system, pointer, size);
            join(system, buffer, network, passphrase, spectator);
        }
        // GetNodeInformationList(network, tag 0, tag 1), the nodes a beacon lists, into the title's static buffer
        0x001F => {
            let pointer = word(system, 2);
            let network = read(system, pointer, NETWORK_SIZE);
            let tag = |system: &mut System, descriptor: u32, pointer: u32| {
                let size = match Descriptor::parse(descriptor) {
                    Descriptor::StaticBuffer { size, .. } => size as usize,
                    _ => 0,
                };
                read(system, pointer, size.min(0x100))
            };
            let (descriptor, pointer) = (word(system, 3), word(system, 4));
            let first = tag(system, descriptor, pointer);
            let (descriptor, pointer) = (word(system, 5), word(system, 6));
            let second = tag(system, descriptor, pointer);
            let nodes = beacon_nodes(&network, &first, &second);
            reply_static(system, buffer, command, &[], &nodes, nodes.len());
        }
        // the rest take settings no console here needs, SetMaxSendDelay, Flush, SetProbeResponseParam,
        // ScanOnConnection and the unnamed ones
        0x0006 | 0x0015 | 0x0016 | 0x0018 | 0x0019 | 0x001C | 0x0020 | 0x0021 | 0x0022 => {
            reply_result(system, buffer, command, 0);
        }
        0x0023 => buffer.reply(&mut system.memory, command, &[0]),
        _ => return false,
    }
    true
}

fn reply_result(system: &mut System, buffer: &CommandBuffer, command: u16, code: u32) {
    if code == 0 {
        buffer.reply(&mut system.memory, command, &[]);
    } else {
        buffer.reply_error(&mut system.memory, command, code);
    }
}

/// replies w/ words and bytes in the title's first static buffer, as many as length, zeroes past them, returns where the bytes went
fn reply_static(system: &mut System, buffer: &CommandBuffer, command: u16, words: &[u32], bytes: &[u8], length: usize) -> u32 {
    let tls = system.kernel.current().map_or(0, |thread| thread.tls);
    let (destination, capacity) = buffer.static_buffer(&mut system.memory, tls, 0);
    let length = length.min(capacity as usize);
    let mut out = vec![0u8; length];
    let copied = bytes.len().min(length);
    out[..copied].copy_from_slice(&bytes[..copied]);
    system.memory.write_bytes(destination, &out);
    let normal = words.len() as u32 + 1;
    buffer.set(&mut system.memory, 0, Header::new(command, normal, 2).0);
    buffer.set(&mut system.memory, 1, 0);
    for (i, &value) in words.iter().enumerate() {
        buffer.set(&mut system.memory, 2 + i as u32, value);
    }
    buffer.set(&mut system.memory, normal + 1, Descriptor::static_buffer(length as u32, 0));
    buffer.set(&mut system.memory, normal + 2, destination);
    destination
}

fn read(system: &mut System, pointer: u32, size: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; size];
    system.memory.read_bytes(pointer, &mut bytes);
    bytes
}

/// the console's own node, w/ the id it has in the network
fn node_bytes(peer: &Peer) -> Vec<u8> {
    let mut node = peer.node.clone();
    node.resize(NODE_SIZE, 0);
    node[0x20..0x22].copy_from_slice(&peer.node_id.to_le_bytes());
    node
}

fn signal_status(system: &mut System) {
    if let Some(event) = system.services.uds.status_event {
        system.kernel.signal_event(event);
    }
}

fn send(system: &System, to: Mac, message: &Wireless) {
    let Some(room) = &system.room else { return };
    match postcard::to_allocvec(message) {
        Ok(body) => room.send(to, body),
        Err(error) => log::warn!("uds: could not write {message:?}, {error}"),
    }
}

/// starts the service (or starts it over) w/ the title's node
fn initialize(system: &mut System, node: Vec<u8>) -> ObjectId {
    leave(system);
    let event = match system.services.uds.status_event {
        Some(event) => event,
        None => {
            let event = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "UDS status")));
            // signaled whenever the status changes, kept whatever the title does w/ its handle
            system.kernel.objects.add_ref(event);
            event
        }
    };
    let uds = &mut system.services.uds;
    uds.status_event = Some(event);
    uds.node = node;
    uds.initialized = true;
    uds.status = Status { status: NOT_CONNECTED, ..Status::default() };
    uds.binds.clear();
    uds.beacons.clear();
    event
}

/// puts a network up, the console is its host and first node
fn host(system: &mut System, mut network: Vec<u8>, passphrase: Vec<u8>) -> u32 {
    let Some(mac) = system.room.as_ref().filter(|room| room.open()).map(|room| room.mac()) else {
        return result(NOT_AUTHORIZED, Summary::InvalidState, Level::Status);
    };
    let max = network[0x1D].clamp(2, MAX_NODES as u8);
    network[0..6].copy_from_slice(&mac);
    if network[6] == 0 {
        network[6] = DEFAULT_CHANNEL;
    }
    network[0x0C..0x0F].copy_from_slice(&NINTENDO_OUI);
    network[0x0F] = 21;
    network[0x1C] = 1;
    network[0x1D] = max;
    network[0x3F] = network[0x3F].min(APPLICATION_DATA as u8);
    let tick = system.cpu.cycles;
    let uds = &mut system.services.uds;
    let me = Peer { node_id: HOST_NODE, mac, node: uds.node.clone() };
    uds.peers = vec![me];
    uds.network = network;
    uds.passphrase = passphrase;
    uds.host = Some(mac);
    uds.status = Status { status: HOST, reason: ESTABLISHED, node_id: HOST_NODE, max, ..Status::default() };
    let peers = uds.peers.clone();
    uds.status.take_peers(&peers);
    uds.next_beacon = Some(tick);
    log::debug!("uds: hosting network {:08X} for up to {max}", uds.network_id());
    signal_status(system);
    0
}

/// asks the host of a network to let the console in, the title waits till it answers or JOIN_NANOS go by
fn join(system: &mut System, buffer: &CommandBuffer, network: Vec<u8>, passphrase: Vec<u8>, spectator: bool) {
    let host: Mac = network[0..6].try_into().unwrap();
    let network_id = u32::from_be_bytes(network[0x18..0x1C].try_into().unwrap());
    // what the title finds if the host never answers
    reply_result(system, buffer, 0x001E, result(TIMEOUT, Summary::Canceled, Level::Status));
    let Some(thread) = system.kernel.current_thread else { return };
    if system.room.as_ref().is_none_or(|room| !room.open()) {
        return;
    }
    let node = system.services.uds.node.clone();
    send(system, host, &Wireless::Join { network_id, spectator, passphrase, node });
    let event = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "UDS join")));
    system.kernel.objects.add_ref(event);
    let tick = system.cpu.cycles;
    let wait = nanos_to_ticks(JOIN_NANOS);
    system.kernel.begin_wait(vec![event], false, Some(wait), tick);
    let uds = &mut system.services.uds;
    uds.network = network;
    uds.host = Some(host);
    uds.status.status = CONNECTING;
    uds.joining = Some(Joining { thread, event, until: tick + wait, spectator });
    log::info!("uds: joining network {network_id:08X} of {}", mac_text(&host));
}

/// leaves the network the console's in, or ends the one it hosts
fn leave(system: &mut System) {
    let uds = &system.services.uds;
    match uds.status.status {
        HOST => {
            let clients: Vec<Mac> = uds.peers.iter().filter(|peer| peer.node_id != HOST_NODE).map(|peer| peer.mac).collect();
            for mac in clients {
                send(system, mac, &Wireless::Leave);
            }
            log::debug!("uds: the network is down");
        }
        CLIENT | SPECTATOR | CONNECTING => {
            if let Some(host) = uds.host {
                send(system, host, &Wireless::Leave);
            }
        }
        _ => {}
    }
    end(system, NOT_CONNECTED, system.services.uds.status.reason);
}

/// out of the network, w/ the reason given, and the title gets told
fn end(system: &mut System, status: u32, reason: u32) {
    let was = system.services.uds.status.status;
    if let Some(joining) = system.services.uds.joining.take() {
        system.kernel.signal_event(joining.event);
        system.kernel.objects.release(joining.event);
    }
    let uds = &mut system.services.uds;
    uds.peers.clear();
    uds.host = None;
    uds.next_beacon = None;
    uds.passphrase.clear();
    for bind in &mut uds.binds {
        bind.queue.clear();
    }
    let node_id = uds.status.node_id;
    uds.status = Status { status, reason, node_id, ..Status::default() };
    if was != status {
        signal_status(system);
    }
}

/// a node leaves the network the console hosts
fn drop_peer(system: &mut System, mac: Mac) {
    let uds = &mut system.services.uds;
    let before = uds.peers.len();
    uds.peers.retain(|peer| peer.mac != mac || peer.node_id == HOST_NODE);
    if uds.peers.len() == before {
        return;
    }
    let peers = uds.peers.clone();
    uds.status.take_peers(&peers);
    let total = uds.status.total;
    if let Some(count) = uds.network.get_mut(0x1C) {
        *count = total;
    }
    tell_peers(system);
    signal_status(system);
}

/// tells the clients who is in the network the console hosts
fn tell_peers(system: &mut System) {
    let peers = system.services.uds.peers.clone();
    for peer in peers.iter().filter(|peer| peer.node_id != HOST_NODE) {
        send(system, peer.mac, &Wireless::Peers(peers.clone()));
    }
}

fn send_to(system: &mut System, to: u16, channel: u8, size: u32, flags: u8, pointer: u32) -> u32 {
    let uds = &system.services.uds;
    if !matches!(uds.status.status, HOST | CLIENT) {
        return result(NOT_AUTHORIZED, Summary::InvalidState, Level::Status);
    }
    if to == 0 || to == uds.status.node_id {
        return result(NOT_FOUND, Summary::WrongArgument, Level::Status);
    }
    if size > MAX_SEND {
        return result(TOO_LARGE, Summary::WrongArgument, Level::Usage);
    }
    let broadcast = to == BROADCAST_NODE || flags & 0x2 != 0;
    let mac = if broadcast {
        BROADCAST
    } else if to == HOST_NODE {
        match uds.host {
            Some(host) => host,
            None => return result(NOT_FOUND, Summary::WrongArgument, Level::Status),
        }
    } else {
        match uds.peers.iter().find(|peer| peer.node_id == to) {
            Some(peer) => peer.mac,
            None => return result(NOT_FOUND, Summary::WrongArgument, Level::Status),
        }
    };
    let message = Wireless::Data {
        network_id: uds.network_id(),
        channel,
        from: uds.status.node_id,
        to: if broadcast { BROADCAST_NODE } else { to },
        payload: read(system, pointer, size as usize),
    };
    send(system, mac, &message);
    0
}

/// writes the beacons heard since the last scan (of any host or of the one asked for) into the title's buffer
fn scan(system: &mut System, pointer: u32, size: u32, wanted: Mac) {
    let uds = &mut system.services.uds;
    let mut taken = Vec::new();
    uds.beacons.retain(|beacon| {
        if wanted == BROADCAST || beacon.mac == wanted {
            taken.push((beacon.mac, beacon.channel, beacon.frame.clone()));
            false
        } else {
            true
        }
    });
    let mut out = vec![0u8; 0xC];
    let mut count = 0u32;
    for (mac, channel, frame) in taken {
        let length = 0x1C + frame.len();
        if out.len() + length > size as usize {
            break;
        }
        let mut entry = vec![0u8; 0x1C];
        entry[0..4].copy_from_slice(&(length as u32).to_le_bytes());
        entry[5] = channel;
        entry[8..14].copy_from_slice(&mac);
        entry[0x14..0x18].copy_from_slice(&(length as u32).to_le_bytes());
        entry[0x18..0x1C].copy_from_slice(&0x1Cu32.to_le_bytes());
        out.extend_from_slice(&entry);
        out.extend_from_slice(&frame);
        count += 1;
    }
    let total = out.len() as u32;
    out[0..4].copy_from_slice(&size.to_le_bytes());
    out[4..8].copy_from_slice(&total.to_le_bytes());
    out[8..12].copy_from_slice(&count.to_le_bytes());
    system.memory.write_bytes(pointer, &out);
}

/// a host's beacon, the way a title reads one out of a scan
/// the fixed fields of an 802.11 beacon, an SSID of eight zeroes, then Nintendo's tags, the network's and its nodes'
/// the nodes go in the clear, only GetNodeInformationList reads them
fn beacon(network: &[u8], peers: &[Peer]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(0x200);
    // up for 15 minutes, every 100 time units, its capabilities
    frame.extend_from_slice(&900_000_000u64.to_le_bytes());
    frame.extend_from_slice(&100u16.to_le_bytes());
    frame.extend_from_slice(&0x0431u16.to_le_bytes());
    frame.extend_from_slice(&[0, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
    frame.extend_from_slice(&[0xDD, 7, 0x00, 0x1F, 0x32, 20, 0x0A, 0x00, 0x00]);

    // the network, from its OUI on, hashed w/ its hash zeroed
    let appdata = (network[0x3F] as usize).min(APPLICATION_DATA);
    let mut info = network[0x0C..0x0C + 0x1F].to_vec();
    info.extend_from_slice(&[0; 0x14]);
    info.push(appdata as u8);
    info.extend_from_slice(&network[0x40..0x40 + appdata]);
    let hash: [u8; 20] = Sha1::digest(&info).into();
    info[0x1F..0x33].copy_from_slice(&hash);
    frame.push(0xDD);
    frame.push(info.len() as u8);
    frame.extend_from_slice(&info);

    // the nodes, host first, big endian like the console has them
    let max = (network[0x1D] as usize).min(MAX_NODES);
    let mut nodes = vec![0u8; 0x12];
    let mask = peers.iter().filter(|peer| (1..=max as u16).contains(&peer.node_id)).fold(0u16, |mask, peer| mask | 1 << (peer.node_id - 1));
    nodes[0x10..0x12].copy_from_slice(&mask.to_be_bytes());
    for index in 0..max {
        let mut entry = [0u8; 0x1E];
        if let Some(peer) = peers.iter().find(|peer| peer.node_id as usize == index + 1) {
            let node = node_bytes(peer);
            entry[0..8].copy_from_slice(&u64::from_le_bytes(node[0..8].try_into().unwrap()).to_be_bytes());
            for i in 0..10 {
                let unit = u16::from_le_bytes([node[8 + i * 2], node[9 + i * 2]]);
                entry[8 + i * 2..10 + i * 2].copy_from_slice(&unit.to_be_bytes());
            }
            entry[0x1C..0x1E].copy_from_slice(&peer.node_id.to_be_bytes());
        }
        nodes.extend_from_slice(&entry);
    }
    for (kind, part) in [(24u8, &nodes[..nodes.len().min(FIRST_TAG_DATA)]), (25, &nodes[nodes.len().min(FIRST_TAG_DATA)..])] {
        if kind == 25 && part.is_empty() {
            break;
        }
        frame.push(0xDD);
        frame.push((4 + part.len()) as u8);
        frame.extend_from_slice(&NINTENDO_OUI);
        frame.push(kind);
        frame.extend_from_slice(part);
    }
    frame
}

/// the nodes a beacon's tags list, sixteen NodeInfo little endian
fn beacon_nodes(network: &[u8], first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut data = first.get(4..).unwrap_or_default().to_vec();
    data.extend_from_slice(second.get(4..).unwrap_or_default());
    let max = (network[0x1D] as usize).min(MAX_NODES);
    let mut out = vec![0u8; NODE_SIZE * MAX_NODES];
    for index in 0..max {
        let Some(entry) = data.get(0x12 + index * 0x1E..0x12 + (index + 1) * 0x1E) else { break };
        let node = &mut out[index * NODE_SIZE..(index + 1) * NODE_SIZE];
        node[0..8].copy_from_slice(&u64::from_be_bytes(entry[0..8].try_into().unwrap()).to_le_bytes());
        for i in 0..10 {
            let unit = u16::from_be_bytes([entry[8 + i * 2], entry[9 + i * 2]]);
            node[8 + i * 2..10 + i * 2].copy_from_slice(&unit.to_le_bytes());
        }
        node[0x20..0x22].copy_from_slice(&u16::from_be_bytes([entry[0x1C], entry[0x1D]]).to_le_bytes());
    }
    out
}

fn mac_text(mac: &Mac) -> String {
    mac.iter().map(|byte| format!("{byte:02X}")).collect::<Vec<_>>().join(":")
}

/// what came thru the room, the beacon due, and a join that took too long
pub fn tick(system: &mut System) {
    let Some(room) = system.room.clone() else { return };
    for (from, body) in room.receive() {
        match postcard::from_bytes::<Wireless>(&body) {
            Ok(message) if message.download_play() => crate::services::dlp::take(system, from, message),
            Ok(message) if system.services.uds.initialized => take(system, from, message),
            Ok(_) => {}
            Err(error) => log::warn!("uds: {} sent something unreadable, {error}", mac_text(&from)),
        }
    }
    if !system.services.uds.initialized {
        return;
    }
    if !room.open() && system.services.uds.status.status != NOT_CONNECTED {
        log::info!("uds: the room is gone, and the network with it");
        end(system, NOT_CONNECTED, LOST);
        return;
    }
    let now = system.cpu.cycles;
    let uds = &mut system.services.uds;
    if let Some(at) = uds.next_beacon.filter(|&at| at <= now) {
        let period = nanos_to_ticks(BEACON_NANOS);
        uds.next_beacon = Some(at + period * ((now - at) / period + 1));
        let message = Wireless::Beacon { channel: uds.network[6], frame: beacon(&uds.network, &uds.peers) };
        send(system, BROADCAST, &message);
    }
    if system.services.uds.joining.as_ref().is_some_and(|joining| joining.until <= now) {
        log::info!("uds: the host never answered");
        end(system, NOT_CONNECTED, system.services.uds.status.reason);
    }
}

/// what a console sent this one
fn take(system: &mut System, from: Mac, message: Wireless) {
    let status = system.services.uds.status.status;
    match message {
        Wireless::Beacon { channel, frame } => {
            if status == HOST {
                return;
            }
            let beacons = &mut system.services.uds.beacons;
            beacons.retain(|beacon| beacon.mac != from);
            if beacons.len() >= MAX_BEACONS {
                beacons.pop_front();
            }
            beacons.push_back(Beacon { mac: from, channel, frame });
        }
        Wireless::Join { network_id, spectator, passphrase, node } => {
            let uds = &system.services.uds;
            if status != HOST {
                return;
            }
            if network_id != uds.network_id() || passphrase != uds.passphrase {
                log::info!("uds: {} asked to join with the wrong network or passphrase", mac_text(&from));
                send(system, from, &Wireless::Refused);
                return;
            }
            let node_id = match uds.peers.iter().find(|peer| peer.mac == from) {
                Some(peer) => peer.node_id,
                None if spectator => 0,
                None => match (2..=uds.status.max as u16).find(|id| uds.peers.iter().all(|peer| peer.node_id != *id)) {
                    Some(id) => id,
                    None => {
                        log::info!("uds: {} asked to join a full network", mac_text(&from));
                        send(system, from, &Wireless::Refused);
                        return;
                    }
                },
            };
            let uds = &mut system.services.uds;
            if !uds.peers.iter().any(|peer| peer.mac == from) && node_id != 0 {
                uds.peers.push(Peer { node_id, mac: from, node });
                let peers = uds.peers.clone();
                uds.status.take_peers(&peers);
                let total = uds.status.total;
                uds.network[0x1C] = total;
            }
            log::info!("uds: {} joined as node {node_id}", mac_text(&from));
            let welcome = Wireless::Welcome { node_id, network: uds.network.clone(), peers: uds.peers.clone() };
            send(system, from, &welcome);
            tell_peers(system);
            signal_status(system);
        }
        Wireless::Welcome { node_id, network, peers } => {
            let uds = &mut system.services.uds;
            let Some(joining) = uds.joining.take() else { return };
            if uds.host != Some(from) {
                uds.joining = Some(joining);
                return;
            }
            uds.network = network;
            uds.peers = peers;
            let (max, peers) = (uds.network.get(0x1D).copied().unwrap_or(0), uds.peers.clone());
            uds.status = Status {
                status: if joining.spectator { SPECTATOR } else { CLIENT },
                reason: ESTABLISHED,
                node_id,
                max,
                ..Status::default()
            };
            uds.status.take_peers(&peers);
            // ConnectNetwork2 succeeds now, and its title goes on
            let tls = system.kernel.thread(joining.thread).tls;
            CommandBuffer::new(tls).reply(&mut system.memory, 0x001E, &[]);
            system.kernel.signal_event(joining.event);
            system.kernel.objects.release(joining.event);
            log::info!("uds: joined as node {node_id}");
            signal_status(system);
        }
        Wireless::Refused => {
            if system.services.uds.joining.is_some() && system.services.uds.host == Some(from) {
                log::info!("uds: the host turned this console away");
                end(system, NOT_CONNECTED, ESTABLISHED);
            }
        }
        Wireless::Peers(peers) => {
            let uds = &mut system.services.uds;
            if !matches!(status, CLIENT | SPECTATOR) || uds.host != Some(from) {
                return;
            }
            uds.peers = peers;
            let peers = uds.peers.clone();
            uds.status.take_peers(&peers);
            signal_status(system);
        }
        Wireless::Leave => match status {
            HOST => drop_peer(system, from),
            CLIENT | SPECTATOR | CONNECTING if system.services.uds.host == Some(from) => {
                log::info!("uds: the host ended the network, or sent this console away");
                end(system, NOT_CONNECTED, LOST);
            }
            _ => {}
        },
        Wireless::Data { network_id, channel, from: node, to, payload } => {
            let uds = &mut system.services.uds;
            // networks may share an id, the sender has to be one of this one's
            let ours = uds.peers.iter().any(|peer| peer.mac == from) || uds.host == Some(from);
            if !uds.connected() || !ours || network_id != uds.network_id() || (to != BROADCAST_NODE && to != uds.status.node_id) {
                return;
            }
            let mut events = Vec::new();
            for bind in uds.binds.iter_mut().filter(|bind| bind.channel == channel && (bind.from == BROADCAST_NODE || bind.from == node)) {
                // a title that stopped pulling doesn't get them piling up
                if bind.queue.len() >= 256 {
                    bind.queue.pop_front();
                }
                bind.queue.push_back((node, payload.clone()));
                events.push(bind.event);
            }
            for event in events {
                system.kernel.signal_event(event);
            }
        }
        // the dlp service's, see tick
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::room::Link;
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};

    /// a system in a room, its title's buffers mapped at 0x0800_0000, plus a static buffer to receive into
    fn console(link: Link) -> (System, CommandBuffer) {
        let (mut system, buffer) = system_with_thread();
        system.room = Some(link);
        let page = system.memory.phys.allocate(crate::memory::MemoryRegion::Application, 0x4000).unwrap();
        system.memory.map(0x0800_0000, page.addr, 0x4000, crate::memory::Permission::RW, crate::memory::MemoryState::Private);
        let tls = system.kernel.current().unwrap().tls;
        system.memory.write_bytes(tls + 0x180, &Descriptor::static_buffer(0x1000, 0).to_le_bytes());
        system.memory.write_bytes(tls + 0x184, &0x0800_3000u32.to_le_bytes());
        (system, buffer)
    }

    fn request(system: &mut System, buffer: &CommandBuffer, words: &[u32]) -> u32 {
        for (i, word) in words.iter().enumerate() {
            buffer.set(&mut system.memory, i as u32, *word);
        }
        handle_request(system, Target::service("nwm::UDS".into(), 0));
        buffer.get(&mut system.memory, 1)
    }

    /// lets what was sent arrive and the consoles take it
    fn settle(consoles: &mut [&mut System]) {
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            for system in consoles.iter_mut() {
                tick(system);
            }
        }
    }

    fn initialize(system: &mut System, buffer: &CommandBuffer, name: char) {
        let mut words = vec![0x001B_0302, 0x1000];
        let mut node = [0u8; 40];
        node[0..8].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        node[8..10].copy_from_slice(&(name as u16).to_le_bytes());
        words.extend(node.chunks(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())));
        words.extend([0x400, 0, 0]);
        assert_eq!(request(system, buffer, &words), 0);
    }

    /// one console hosts a network, the other finds it scanning, reads its nodes, joins w/ the passphrase and the two pass data both ways
    #[test]
    fn two_consoles_meet_and_talk() {
        let (first, second) = Link::pair();
        let (mut host, host_buffer) = console(first);
        let (mut client, client_buffer) = console(second);
        initialize(&mut host, &host_buffer, 'H');
        initialize(&mut client, &client_buffer, 'C');

        // the network, comm id 0xABCD, id 7, network id 0x11223344, up to 4 nodes
        let mut network = vec![0u8; NETWORK_SIZE];
        network[0x10..0x14].copy_from_slice(&0xABCDu32.to_be_bytes());
        network[0x14] = 7;
        network[0x18..0x1C].copy_from_slice(&0x1122_3344u32.to_be_bytes());
        network[0x1D] = 4;
        host.memory.write_bytes(0x0800_0000, &network);
        host.memory.write_bytes(0x0800_0200, b"password");
        let passphrase = Descriptor::static_buffer(8, 0);
        assert_eq!(request(&mut host, &host_buffer, &[0x001D_0044, 8, Descriptor::static_buffer(0x108, 1), 0x0800_0000, passphrase, 0x0800_0200]), 0);
        assert_eq!(host.services.uds.status.status, HOST);

        // a beacon goes out and a scan finds it
        settle(&mut [&mut host, &mut client]);
        let mut scan = vec![0x000F_0404, 0x1000];
        scan.extend([0u32; 13]);
        scan[4] = 0xFFFF_FFFF;
        scan[5] = 0xFFFF;
        scan.extend([0xABCD, 7, 0, 0, (0x1000 << 4) | 12, 0x0800_1000]);
        assert_eq!(request(&mut client, &client_buffer, &scan), 0);
        let mut reply = vec![0u8; 0x400];
        client.memory.read_bytes(0x0800_1000, &mut reply);
        assert_eq!(u32::from_le_bytes(reply[8..12].try_into().unwrap()), 1, "one beacon");
        let frame = &reply[0xC + 0x1C..];
        // the network's tag, after the fixed fields, the SSID and the first tag
        let info = &frame[12 + 10 + 9..];
        assert_eq!(&info[0..2], &[0xDD, 0x34]);
        assert_eq!(&info[2..6], &[0x00, 0x1F, 0x32, 21]);
        assert_eq!(&info[6..10], &0xABCDu32.to_be_bytes());
        let mut hashed = info[2..2 + 0x34].to_vec();
        hashed[0x1F..0x33].fill(0);
        let hash: [u8; 20] = Sha1::digest(&hashed).into();
        assert_eq!(&info[2 + 0x1F..2 + 0x33], &hash);

        // the beacon's nodes, thru GetNodeInformationList
        let tag0 = &info[2 + 0x34..];
        assert_eq!(tag0[0], 0xDD);
        let tag0 = &tag0[2..2 + tag0[1] as usize];
        client.memory.write_bytes(0x0800_0400, tag0);
        let mut seen = network.clone();
        seen[0..6].copy_from_slice(&host.room.as_ref().unwrap().mac());
        client.memory.write_bytes(0x0800_0000, &seen);
        let words = [
            0x001F_0006,
            Descriptor::static_buffer(0x108, 1),
            0x0800_0000,
            Descriptor::static_buffer(tag0.len() as u32, 2),
            0x0800_0400,
            Descriptor::static_buffer(0, 3),
            0x0800_0600,
        ];
        assert_eq!(request(&mut client, &client_buffer, &words), 0);
        let mut nodes = [0u8; 0x80];
        client.memory.read_bytes(0x0800_3000, &mut nodes);
        assert_eq!(u64::from_le_bytes(nodes[0..8].try_into().unwrap()), 0x1234_5678);
        assert_eq!(u16::from_le_bytes([nodes[8], nodes[9]]), 'H' as u16);
        assert_eq!(u16::from_le_bytes([nodes[0x20], nodes[0x21]]), 1);

        // a wrong passphrase gets turned away
        client.memory.write_bytes(0x0800_0200, b"wrongpass");
        let wrong = [0x001E_0084, 1, 9, Descriptor::static_buffer(0x108, 1), 0x0800_0000, Descriptor::static_buffer(9, 0), 0x0800_0200];
        request(&mut client, &client_buffer, &wrong);
        settle(&mut [&mut host, &mut client]);
        assert_eq!(client.services.uds.status.status, NOT_CONNECTED);
        assert_ne!(client_buffer.get(&mut client.memory, 1), 0);

        // and the right one let in, as node 2
        client.memory.write_bytes(0x0800_0200, b"password");
        let connect = [0x001E_0084, 1, 8, Descriptor::static_buffer(0x108, 1), 0x0800_0000, passphrase, 0x0800_0200];
        request(&mut client, &client_buffer, &connect);
        assert_eq!(client.services.uds.status.status, CONNECTING);
        settle(&mut [&mut host, &mut client]);
        assert_eq!(client.services.uds.status.status, CLIENT);
        assert_eq!(client.services.uds.status.node_id, 2);
        assert_eq!(client_buffer.get(&mut client.memory, 1), 0, "ConnectNetwork2 succeeded");
        assert_eq!(host.services.uds.status.total, 2);
        let status = host.services.uds.status.words();
        assert_eq!(status[0], HOST);
        assert_eq!(status[11] >> 16, 0b11, "nodes 1 and 2");

        // both bind channel 1 and send to each other
        for (system, buffer) in [(&mut host, &host_buffer), (&mut client, &client_buffer)] {
            assert_eq!(request(system, buffer, &[0x0012_0100, 1, 0x5F4, 1, BROADCAST_NODE as u32]), 0);
        }
        client.memory.write_bytes(0x0800_0800, b"from the client");
        assert_eq!(request(&mut client, &client_buffer, &[0x0017_0182, 0, 1, 1, 4, 15, 0, Descriptor::static_buffer(15, 0), 0x0800_0800]), 0);
        host.memory.write_bytes(0x0800_0800, b"to everyone");
        assert_eq!(request(&mut host, &host_buffer, &[0x0017_0182, 0, BROADCAST_NODE as u32, 1, 3, 11, 0, Descriptor::static_buffer(11, 0), 0x0800_0800]), 0);
        settle(&mut [&mut host, &mut client]);
        let pull = |system: &mut System, buffer: &CommandBuffer| {
            assert_eq!(request(system, buffer, &[0x0014_00C0, 1, 0x172, 0x5C8]), 0);
            let (size, from) = (buffer.get(&mut system.memory, 2), buffer.get(&mut system.memory, 3));
            let mut data = vec![0u8; size as usize];
            system.memory.read_bytes(0x0800_3000, &mut data);
            (data, from)
        };
        assert_eq!(pull(&mut host, &host_buffer), (b"from the client".to_vec(), 2));
        assert_eq!(pull(&mut client, &client_buffer), (b"to everyone".to_vec(), 1));
        assert_eq!(pull(&mut client, &client_buffer).0, Vec::<u8>::new(), "nothing more");

        // the client leaves, and the host sees it go
        assert_eq!(request(&mut client, &client_buffer, &[0x000A_0000]), 0);
        settle(&mut [&mut host, &mut client]);
        assert_eq!(host.services.uds.status.total, 1);
        assert_eq!(client.services.uds.status.status, NOT_CONNECTED);
    }

    /// w/o a room wireless is off like it was
    #[test]
    fn no_room_is_wireless_off() {
        let (mut system, buffer) = system_with_thread();
        let mut words = vec![0x001B_0302, 0x1000];
        words.extend([0u32; 10]);
        words.extend([0x400, 0, 0]);
        assert_eq!(request(&mut system, &buffer, &words), zakuro_common::result::errors::UDS_WIRELESS_OFF.0);
    }
}
