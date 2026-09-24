//! IPC command buffers.

use zakuro_common::memory_map::{TLS_IPC_COMMAND_BUFFER, TLS_IPC_STATIC_BUFFERS};
use zakuro_common::VAddr;

use crate::memory::Memory;

/// the header word at the start of every command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header(pub u32);

impl Header {
    pub const fn new(command_id: u16, normal_params: u32, translate_params: u32) -> Header {
        Header(
            ((command_id as u32) << 16) | ((normal_params & 0x3F) << 6) | (translate_params & 0x3F),
        )
    }

    pub const fn command_id(self) -> u16 {
        (self.0 >> 16) as u16
    }

    pub const fn normal_params(self) -> u32 {
        (self.0 >> 6) & 0x3F
    }

    pub const fn translate_params(self) -> u32 {
        self.0 & 0x3F
    }
}

/// a parsed translate descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Descriptor {
    /// one or more handles follow.
    Handles { count: u32, close: bool },
    /// the sender's process handle is substituted in.
    CurrentProcessId,
    /// points at one of the receiver's static buffers.
    StaticBuffer { size: u32, buffer_id: u32 },
    /// a read/write window into the sender's address space.
    MappedBuffer { size: u32, permission: u32 },
    Unknown(u32),
}

impl Descriptor {
    pub fn parse(raw: u32) -> Descriptor {
        match raw & 0xF {
            0x0 => {
                if raw & 0x20 != 0 {
                    Descriptor::CurrentProcessId
                } else {
                    Descriptor::Handles {
                        count: ((raw >> 26) & 0x3F) + 1,
                        close: raw & 0x10 != 0,
                    }
                }
            }
            0x2 => Descriptor::StaticBuffer {
                size: (raw >> 14) & 0x3FFFF,
                buffer_id: (raw >> 10) & 0xF,
            },
            0x8 | 0xA | 0xC => Descriptor::MappedBuffer {
                size: raw >> 4,
                permission: raw & 0xF,
            },
            _ => Descriptor::Unknown(raw),
        }
    }

    /// a descriptor that copies handles, the receiver gets its own, and the
    /// sender keeps its copies.
    pub const fn handles(count: u32) -> u32 {
        (count - 1) << 26
    }

    pub const fn move_handles(count: u32) -> u32 {
        ((count - 1) << 26) | 0x10
    }

    pub const fn static_buffer(size: u32, buffer_id: u32) -> u32 {
        (size << 14) | (buffer_id << 10) | 0x2
    }
}

/// read/write access to a thread's command buffer.
pub struct CommandBuffer {
    base: VAddr,
}

impl CommandBuffer {
    pub fn new(tls: VAddr) -> CommandBuffer {
        CommandBuffer {
            base: tls + TLS_IPC_COMMAND_BUFFER,
        }
    }

    pub fn address(&self) -> VAddr {
        self.base
    }

    pub fn get(&self, memory: &mut Memory, index: u32) -> u32 {
        use zakuro_cpu::Bus;
        memory.read32(self.base + index * 4)
    }

    pub fn set(&self, memory: &mut Memory, index: u32, value: u32) {
        use zakuro_cpu::Bus;
        memory.write32(self.base + index * 4, value);
    }

    pub fn header(&self, memory: &mut Memory) -> Header {
        Header(self.get(memory, 0))
    }

    /// writes a standard reply, the same command id, values.len() normal
    /// parameters, and a leading result code of zero.
    pub fn reply(&self, memory: &mut Memory, command_id: u16, values: &[u32]) {
        self.set(
            memory,
            0,
            Header::new(command_id, values.len() as u32 + 1, 0).0,
        );
        self.set(memory, 1, 0); // success
        for (i, &value) in values.iter().enumerate() {
            self.set(memory, i as u32 + 2, value);
        }
    }

    /// writes a reply carrying an error code and nothing else.
    pub fn reply_error(&self, memory: &mut Memory, command_id: u16, code: u32) {
        self.set(memory, 0, Header::new(command_id, 1, 0).0);
        self.set(memory, 1, code);
    }

    /// writes a reply whose translate section moves a handle to the caller.
    pub fn reply_with_handle(&self, memory: &mut Memory, command_id: u16, handle: u32) {
        self.set(memory, 0, Header::new(command_id, 1, 2).0);
        self.set(memory, 1, 0);
        self.set(memory, 2, Descriptor::move_handles(1));
        self.set(memory, 3, handle);
    }

    /// the static buffer descriptor the receiver published for slot id.
    pub fn static_buffer(&self, memory: &mut Memory, tls: VAddr, id: u32) -> (VAddr, u32) {
        use zakuro_cpu::Bus;
        let base = tls + TLS_IPC_STATIC_BUFFERS + id * 8;
        let descriptor = memory.read32(base);
        let pointer = memory.read32(base + 4);
        (pointer, (descriptor >> 14) & 0x3FFFF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fields_round_trip() {
        let h = Header::new(0x0001, 2, 2);
        assert_eq!(h.command_id(), 1);
        assert_eq!(h.normal_params(), 2);
        assert_eq!(h.translate_params(), 2);
        // this is the exact header libctru builds for srv:'s GetServiceHandle.
        assert_eq!(Header::new(0x0005, 4, 0).0, 0x0005_0100);
    }

    #[test]
    fn parses_descriptors() {
        assert_eq!(
            Descriptor::parse(Descriptor::move_handles(1)),
            Descriptor::Handles {
                count: 1,
                close: true
            }
        );
        assert_eq!(
            Descriptor::parse(Descriptor::static_buffer(0x20, 0)),
            Descriptor::StaticBuffer {
                size: 0x20,
                buffer_id: 0
            }
        );
        assert_eq!(
            Descriptor::parse(0x0000_000C | (0x100 << 4)),
            Descriptor::MappedBuffer {
                size: 0x100,
                permission: 0xC
            }
        );
    }
}
