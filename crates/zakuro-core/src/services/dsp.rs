//! dsp::DSP, the audio DSP.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::ObjectId;
use crate::kernel::sync::ResetType;
use crate::System;

/// command ids from the DSP service table.
mod command {
    pub const RECV_DATA: u16 = 0x0001;
    pub const RECV_DATA_IS_READY: u16 = 0x0002;
    pub const SEND_DATA: u16 = 0x0003;
    pub const SEND_DATA_IS_EMPTY: u16 = 0x0004;
    pub const SEND_FIFO_EX: u16 = 0x0005;
    pub const RECV_FIFO_EX: u16 = 0x0006;
    pub const SET_SEMAPHORE: u16 = 0x0007;
    pub const GET_SEMAPHORE: u16 = 0x0008;
    pub const CLEAR_SEMAPHORE: u16 = 0x0009;
    pub const MASK_SEMAPHORE: u16 = 0x000A;
    pub const CHECK_SEMAPHORE_REQUEST: u16 = 0x000B;
    pub const CONVERT_PROCESS_ADDRESS_FROM_DSP_DRAM: u16 = 0x000C;
    pub const WRITE_PROCESS_PIPE: u16 = 0x000D;
    pub const READ_PIPE: u16 = 0x000E;
    pub const GET_PIPE_READABLE_SIZE: u16 = 0x000F;
    pub const READ_PIPE_IF_POSSIBLE: u16 = 0x0010;
    pub const LOAD_COMPONENT: u16 = 0x0011;
    pub const UNLOAD_COMPONENT: u16 = 0x0012;
    pub const FLUSH_DATA_CACHE: u16 = 0x0013;
    pub const INVALIDATE_DATA_CACHE: u16 = 0x0014;
    pub const REGISTER_INTERRUPT_EVENTS: u16 = 0x0015;
    pub const GET_SEMAPHORE_EVENT_HANDLE: u16 = 0x0016;
    pub const SET_SEMAPHORE_MASK: u16 = 0x0017;
    pub const GET_HEADPHONE_STATUS: u16 = 0x0018;
    pub const FORCE_HEADPHONE_OUT: u16 = 0x0019;
    pub const GET_IS_DSP_OCCUPIED: u16 = 0x001A;
}

/// the pipe a title uses to drive the audio firmware.
const PIPE_AUDIO: u32 = 2;
const PIPE_COUNT: usize = 8;

/// what the title writes to the audio pipe to change the firmware's state.
const STATE_INITIALIZE: u8 = 0;
const STATE_SHUTDOWN: u8 = 1;
const STATE_WAKEUP: u8 = 2;
const STATE_SLEEP: u8 = 3;

/// addresses in DSP data memory of the structures the firmware exposes.
mod layout {
    pub const FRAME_COUNTER: u16 = 0xBFFF;
    pub const SOURCE_CONFIGURATIONS: u16 = 0x9E92;
    pub const SOURCE_STATUSES: u16 = 0x8680;
    pub const ADPCM_COEFFICIENTS: u16 = 0xA792;
    pub const DSP_CONFIGURATION: u16 = 0x9430;
    pub const DSP_STATUS: u16 = 0x8400;
    pub const FINAL_SAMPLES: u16 = 0x8540;
    pub const INTERMEDIATE_MIX_SAMPLES: u16 = 0x9492;
    pub const COMPRESSOR: u16 = 0x8710;
    pub const DSP_DEBUG: u16 = 0x8410;
    pub const UNKNOWN_10: u16 = 0xA912;
    pub const UNKNOWN_11: u16 = 0xAA12;
    pub const UNKNOWN_12: u16 = 0xAAD2;
    pub const UNKNOWN_13: u16 = 0xAC52;
    pub const UNKNOWN_14: u16 = 0xAC5C;
}

#[derive(Default)]
pub struct DspState {
    pub component_loaded: bool,
    pub running: bool,
    pub semaphore_event: Option<u32>,
    pub semaphore_event_object: Option<ObjectId>,
    /// events registered for the three DSP interrupts.
    pub interrupt_events: Vec<ObjectId>,
    pub semaphore_mask: u32,
    /// what GetSemaphore reports.
    pub semaphore_value: u32,
    /// data waiting to be read out of each pipe.
    pipes: [Vec<u8>; PIPE_COUNT],
    /// the 24 voices the audio firmware plays.
    voices: super::dsp_voices::Voices,
}

impl DspState {
    fn reset_pipes(&mut self) {
        for pipe in &mut self.pipes {
            pipe.clear();
        }
    }

    /// queues the reply the real firmware sends after initialization, the
    /// count of structures it exposes, then where each one lives.
    fn write_struct_addresses(&mut self) {
        const ADDRESSES: [u16; 15] = [
            layout::FRAME_COUNTER,
            layout::SOURCE_CONFIGURATIONS,
            layout::SOURCE_STATUSES,
            layout::ADPCM_COEFFICIENTS,
            layout::DSP_CONFIGURATION,
            layout::DSP_STATUS,
            layout::FINAL_SAMPLES,
            layout::INTERMEDIATE_MIX_SAMPLES,
            layout::COMPRESSOR,
            layout::DSP_DEBUG,
            layout::UNKNOWN_10,
            layout::UNKNOWN_11,
            layout::UNKNOWN_12,
            layout::UNKNOWN_13,
            layout::UNKNOWN_14,
        ];

        let mut response = Vec::with_capacity(2 + ADDRESSES.len() * 2);
        response.extend_from_slice(&(ADDRESSES.len() as u16).to_le_bytes());
        for address in ADDRESSES {
            response.extend_from_slice(&address.to_le_bytes());
        }
        self.pipes[PIPE_AUDIO as usize] = response;
    }

    fn read_pipe(&mut self, pipe: u32, length: usize) -> Vec<u8> {
        let Some(data) = self.pipes.get_mut(pipe as usize) else {
            return Vec::new();
        };
        let count = length.min(data.len());
        data.drain(..count).collect()
    }

    fn readable(&self, pipe: u32) -> usize {
        self.pipes.get(pipe as usize).map_or(0, |d| d.len())
    }
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    use command::*;
    let id = header.command_id();
    match id {
        // RecvData(register) -> the DSP's reply.
        RECV_DATA => {
            let value = if system.services.dsp.running { 0 } else { 1 };
            buffer.reply(&mut system.memory, id, &[value]);
            true
        }
        RECV_DATA_IS_READY => {
            buffer.reply(&mut system.memory, id, &[1]);
            true
        }
        SEND_DATA | SEND_FIFO_EX | SET_SEMAPHORE | CLEAR_SEMAPHORE | MASK_SEMAPHORE => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        SEND_DATA_IS_EMPTY | CHECK_SEMAPHORE_REQUEST => {
            buffer.reply(&mut system.memory, id, &[1]);
            true
        }
        GET_SEMAPHORE => {
            buffer.reply(&mut system.memory, id, &[system.services.dsp.semaphore_value]);
            true
        }
        RECV_FIFO_EX => {
            buffer.reply(&mut system.memory, id, &[0]);
            true
        }
        // ConvertProcessAddressFromDspDram(address) -> the same address in the
        // caller's space. DSP memory is mapped one to one for us.
        CONVERT_PROCESS_ADDRESS_FROM_DSP_DRAM => {
            let address = buffer.get(&mut system.memory, 1);
            let converted =
                zakuro_common::memory_map::DSP_RAM_VADDR + (address << 1) + 0x40000;
            buffer.reply(&mut system.memory, id, &[converted]);
            true
        }
        // WriteProcessPipe(channel, size, <buffer>)
        WRITE_PROCESS_PIPE => {
            let channel = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let source = buffer.get(&mut system.memory, 4);
            let mut data = vec![0u8; size.min(0x1000) as usize];
            system.memory.read_bytes(source, &mut data);
            write_pipe(system, channel, &data);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // ReadPipe / ReadPipeIfPossible(channel, peer, size)
        READ_PIPE | READ_PIPE_IF_POSSIBLE => {
            let channel = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 3) & 0xFFFF;

            let tls = system.kernel.current().map_or(0, |t| t.tls);
            let (destination, capacity) = buffer.static_buffer(&mut system.memory, tls, 0);
            let wanted = size.min(capacity.max(size)) as usize;

            let data = system.services.dsp.read_pipe(channel, wanted);
            let count = data.len() as u32;
            if destination != 0 && !data.is_empty() {
                system.memory.write_bytes(destination, &data);
            }
            log::debug!(
                "dsp: pipe {channel} gave {count} of {size} bytes to 0x{destination:08X}"
            );

            buffer.set(&mut system.memory, 0, Header::new(id, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, count);
            buffer.set(&mut system.memory, 3, Descriptor::static_buffer(count, 0));
            buffer.set(&mut system.memory, 4, destination);
            true
        }

        // GetPipeReadableSize(channel, peer)
        GET_PIPE_READABLE_SIZE => {
            let channel = buffer.get(&mut system.memory, 1);
            let size = system.services.dsp.readable(channel) as u32;
            buffer.reply(&mut system.memory, id, &[size]);
            true
        }
        // LoadComponent(size, program mask, data mask, <buffer>)
        LOAD_COMPONENT => {
            system.services.dsp.component_loaded = true;
            log::info!("dsp: the title uploaded a firmware component; using the HLE audio path");
            // the second word says whether the component was accepted.
            buffer.set(&mut system.memory, 0, Header::new(id, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, 1);
            buffer.set(&mut system.memory, 3, (0x100000 << 4) | 0xA);
            buffer.set(&mut system.memory, 4, 0);
            true
        }
        UNLOAD_COMPONENT => {
            system.services.dsp.component_loaded = false;
            system.services.dsp.running = false;
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        FLUSH_DATA_CACHE | INVALIDATE_DATA_CACHE => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // RegisterInterruptEvents(interrupt, channel, event handle)
        REGISTER_INTERRUPT_EVENTS => {
            let handle = buffer.get(&mut system.memory, 4);
            if let Some(object) = system.kernel.resolve(handle) {
                system.services.dsp.interrupt_events.push(object);
            }
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // GetSemaphoreEventHandle -> the event the audio thread waits on.
        GET_SEMAPHORE_EVENT_HANDLE => {
            let handle = match system.services.dsp.semaphore_event {
                Some(handle) => handle,
                None => {
                    let (object, handle) =
                        system.kernel.create_event(ResetType::OneShot, "DSP:semaphore");
                    system.services.dsp.semaphore_event = Some(handle);
                    system.services.dsp.semaphore_event_object = Some(object);
                    handle
                }
            };
            buffer.set(&mut system.memory, 0, Header::new(id, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
            true
        }
        SET_SEMAPHORE_MASK => {
            system.services.dsp.semaphore_mask = buffer.get(&mut system.memory, 1);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // no headphones are plugged into an emulator.
        GET_HEADPHONE_STATUS => {
            buffer.reply(&mut system.memory, id, &[0]);
            true
        }
        FORCE_HEADPHONE_OUT => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        GET_IS_DSP_OCCUPIED => {
            buffer.reply(&mut system.memory, id, &[0]);
            true
        }
        _ => false,
    }
}

/// handles a write to a pipe.
fn write_pipe(system: &mut System, channel: u32, data: &[u8]) {
    if channel != PIPE_AUDIO || data.is_empty() {
        return;
    }
    match data[0] {
        STATE_INITIALIZE => {
            system.services.dsp.reset_pipes();
            system.services.dsp.write_struct_addresses();
            system.services.dsp.running = true;
            log::debug!("dsp: initialized");
        }
        STATE_SHUTDOWN => {
            system.services.dsp.reset_pipes();
            system.services.dsp.running = false;
            log::debug!("dsp: shut down");
        }
        STATE_WAKEUP => system.services.dsp.running = true,
        STATE_SLEEP => system.services.dsp.running = false,
        other => log::debug!("dsp: unknown audio pipe state change {other}"),
    }
    // the firmware answers every state change by signalling the semaphore.
    signal_semaphore(system);
}

/// converts a DSP data-memory address into the address the ARM side sees.
fn dsp_dram_to_arm(address: u16) -> u32 {
    zakuro_common::memory_map::DSP_RAM_VADDR + ((address as u32) << 1) + 0x40000
}

/// runs one audio frame of the firmware, plays the voices through their
/// buffers and publishes their statuses, which is how a title's sound
/// library knows its audio is moving.
pub fn advance(system: &mut System) {
    if !system.services.dsp.running {
        return;
    }
    let layout = super::dsp_voices::Layout {
        configurations: dsp_dram_to_arm(layout::SOURCE_CONFIGURATIONS),
        statuses: dsp_dram_to_arm(layout::SOURCE_STATUSES),
        frame_counter: dsp_dram_to_arm(layout::FRAME_COUNTER),
    };
    system.services.dsp.voices.tick(&mut system.memory, layout);
}

/// signals the DSP's interrupt events, as the firmware does at the end of
/// every audio frame.
pub fn signal_semaphore(system: &mut System) {
    // real firmware always reports the requested mask back as "satisfied" when
    // it signals the event, a driver that reads 0 here after waking up sees a
    // spurious wakeup and goes straight back to sleep.
    system.services.dsp.semaphore_value = system.services.dsp.semaphore_mask;

    if let Some(object) = system.services.dsp.semaphore_event_object {
        system.kernel.signal_event(object);
    }
    let events = system.services.dsp.interrupt_events.clone();
    for object in events {
        system.kernel.signal_event(object);
    }
}

#[cfg(test)]
mod tests {
    use super::command::*;
    use super::*;
    use crate::kernel::ipc::Header;
    use crate::{Config, System};

    /// the command ids are pinned by the exact headers a title sends.
    #[test]
    fn command_ids_match_the_headers_titles_send() {
        assert_eq!(Header::new(LOAD_COMPONENT, 3, 2).0, 0x0011_00C2);
        assert_eq!(Header::new(REGISTER_INTERRUPT_EVENTS, 2, 2).0, 0x0015_0082);
        assert_eq!(Header::new(GET_SEMAPHORE_EVENT_HANDLE, 0, 0).0, 0x0016_0000);
        assert_eq!(Header::new(WRITE_PROCESS_PIPE, 2, 2).0, 0x000D_0082);
        assert_eq!(Header::new(READ_PIPE_IF_POSSIBLE, 3, 2).0, 0x0010_00C2);
    }

    /// real firmware reports the mask a driver registered with
    /// SetSemaphoreMask back as satisfied whenever it fires the semaphore
    /// event, and a driver that reads back 0 here treats the wakeup as
    /// spurious.
    #[test]
    fn signalling_the_semaphore_reports_the_registered_mask() {
        let mut system = System::new(Config::default());
        system.services.dsp.semaphore_mask = 0x2FFF;
        assert_eq!(system.services.dsp.semaphore_value, 0);

        signal_semaphore(&mut system);

        assert_eq!(system.services.dsp.semaphore_value, 0x2FFF);
    }
}
