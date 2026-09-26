//! high-level service emulation.

pub mod apt;
pub mod err;
pub mod cfg;
pub mod dsp;
pub mod dsp_voices;
pub mod fs;
pub mod glyphs;
pub mod gsp;
pub mod hid;
pub mod ldr_ro;
pub mod misc;
pub mod shared_font;
pub mod host_archive;
pub mod srv;
pub mod system_archives;

use std::collections::BTreeMap;

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

/// where a request is headed.
#[derive(Debug, Clone)]
pub enum Target {
    /// a connection made with svcConnectToPort, e.g. srv:.
    Port(String),
    /// a session obtained from srv:GetServiceHandle.
    Service { name: String, subhandle: u32 },
}

impl Target {
    pub fn port(name: String) -> Target {
        Target::Port(name)
    }

    pub fn service(name: String, subhandle: u32) -> Target {
        Target::Service { name, subhandle }
    }

    pub fn name(&self) -> &str {
        match self {
            Target::Port(name) => name,
            Target::Service { name, .. } => name,
        }
    }
}

/// per-service state that outlives a single request.
#[derive(Default)]
pub struct ServiceState {
    pub srv: srv::SrvState,
    pub apt: apt::AptState,
    pub gsp: gsp::GspState,
    pub hid: hid::HidState,
    pub fs: fs::FsState,
    pub dsp: dsp::DspState,
    /// commands we logged as unimplemented, so the log stays readable and the
    /// diagnostics overlay can show what a title is actually asking for.
    pub unimplemented: BTreeMap<(String, u16), u32>,
}

/// entry point from svcSendSyncRequest.
pub fn handle_request(system: &mut System, target: Target) {
    let Some(thread) = system.kernel.current() else {
        return;
    };
    let buffer = CommandBuffer::new(thread.tls);
    let header = buffer.header(&mut system.memory);
    let command = header.command_id();
    let name = target.name().to_owned();

    log::trace!(
        "IPC {name} cmd 0x{command:04X} ({} normal, {} translate)",
        header.normal_params(),
        header.translate_params()
    );

    let handled = match name.as_str() {
        "srv:" | "srv:pm" => srv::handle(system, &buffer, header),
        "APT:U" | "APT:A" | "APT:S" => apt::handle(system, &buffer, header),
        "gsp::Gpu" => gsp::handle(system, &buffer, header),
        "hid:USER" | "hid:SPVR" => hid::handle(system, &buffer, header),
        "fs:USER" | "FSFile" | "FSDirectory" => fs::handle(system, &buffer, header, &target),
        "cfg:u" | "cfg:s" | "cfg:i" => cfg::handle(system, &buffer, header),
        "err:f" => err::handle(system, &buffer, header),
        "dsp::DSP" => dsp::handle(system, &buffer, header),
        "ldr:ro" => ldr_ro::handle(system, &buffer, header),
        _ => misc::handle(system, &buffer, header, &name),
    };

    if !handled {
        if is_network_service(&name) {
            offline(system, &buffer, header, &name);
        } else {
            unimplemented(system, &buffer, header, &name);
        }
    }
}

/// services whose entire job is talking to the internet or to another console.
fn is_network_service(name: &str) -> bool {
    matches!(
        name,
        "frd:u" | "frd:a"
            | "boss:U" | "boss:P"
            | "nwm::UDS"
            | "http:C"
            | "ssl:C"
            | "nim:aoc" | "nim:s" | "nim:u"
            | "ac:u" | "ac:i"
            | "olv:u"
            | "act:u" | "act:a"
    )
}

/// reply used for a network service we have not implemented, an explicit "not
/// connected" rather than [unimplemented]'s success-with-zeroes.
fn offline(system: &mut System, buffer: &CommandBuffer, header: Header, service: &str) {
    use zakuro_common::result::errors;
    let command = header.command_id();
    let (name, code) = match service {
        // a player can switch wireless off at any time, so a title knows
        // exactly what local wireless says then, a code from another module
        // is just a failure it never expected.
        "nwm::UDS" => ("WIRELESS_OFF", errors::UDS_WIRELESS_OFF),
        _ => ("NOT_CONNECTED", errors::NOT_CONNECTED),
    };
    let count = system
        .services
        .unimplemented
        .entry((service.to_owned(), command))
        .or_insert(0);
    *count += 1;
    if *count == 1 {
        log::warn!(
            "network service {service} command 0x{command:04X}: no network, replying \
             {name} ({code}) instead of stubbing success"
        );
    }
    buffer.reply_error(&mut system.memory, command, code.0);
}

/// default reply for a command we do not implement, success and zeroes.
pub fn unimplemented(system: &mut System, buffer: &CommandBuffer, header: Header, service: &str) {
    let command = header.command_id();
    let count = system
        .services
        .unimplemented
        .entry((service.to_owned(), command))
        .or_insert(0);
    *count += 1;
    if *count == 1 {
        let mut args = Vec::new();
        for i in 1..=header.normal_params().min(8) {
            args.push(format!("0x{:08X}", buffer.get(&mut system.memory, i)));
        }
        log::warn!(
            "unimplemented {service} command 0x{command:04X}({})",
            args.join(", ")
        );
    }
    // reply with one result word of success and nothing else.
    buffer.reply(&mut system.memory, command, &[]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    /// a system with one running thread, whose TLS holds the command buffer
    /// a request is read from and replied into.
    fn system_with_thread() -> (System, CommandBuffer) {
        let mut system = System::new(Config::default());
        let id = system
            .kernel
            .create_thread("main", 0x0010_0000, 0x1000_0000, 0, 0x30, 0);
        system.map_tls_page(id);
        system.kernel.current_thread = Some(id);
        let buffer = CommandBuffer::new(system.kernel.thread(id).tls);
        (system, buffer)
    }

    /// local wireless has to fail the way it does with wireless switched
    /// off, the one failure every title is written to cope with.
    #[test]
    fn local_wireless_reports_wireless_off() {
        let (mut system, buffer) = system_with_thread();
        // InitializeWithVersion, the header the SDK sends.
        buffer.set(&mut system.memory, 0, Header::new(0x001B, 12, 2).0);
        assert_eq!(buffer.get(&mut system.memory, 0), 0x001B_0302);

        handle_request(&mut system, Target::service("nwm::UDS".into(), 0));

        assert_eq!(buffer.header(&mut system.memory), Header::new(0x001B, 1, 0));
        assert_eq!(buffer.get(&mut system.memory, 1), 0xC941_1002);
    }

    /// the other network services keep their explicit "not connected".
    #[test]
    fn other_network_services_report_not_connected() {
        let (mut system, buffer) = system_with_thread();
        buffer.set(&mut system.memory, 0, Header::new(0x0001, 0, 0).0);

        handle_request(&mut system, Target::service("frd:u".into(), 0));

        assert_eq!(
            buffer.get(&mut system.memory, 1),
            zakuro_common::result::errors::NOT_CONNECTED.0
        );
    }

    /// titles answer sleep queries nothing sent them, which has to succeed
    /// quietly.
    #[test]
    fn sleep_query_replies_are_acknowledged() {
        let (mut system, buffer) = system_with_thread();
        // ReplySleepQuery(application, reject)
        buffer.set(&mut system.memory, 0, Header::new(0x003E, 2, 0).0);
        buffer.set(&mut system.memory, 1, 0x300);
        buffer.set(&mut system.memory, 2, 0);

        handle_request(&mut system, Target::service("APT:A".into(), 0));

        assert_eq!(buffer.header(&mut system.memory), Header::new(0x003E, 1, 0));
        assert_eq!(buffer.get(&mut system.memory, 1), 0);
        assert!(system.services.unimplemented.is_empty());
    }

    /// a library applet nothing can run still answers the way one would, a
    /// title waiting on it would otherwise wait forever.
    #[test]
    fn library_applets_answer_and_close() {
        use crate::kernel::ipc::Descriptor;
        use zakuro_common::memory_map::TLS_IPC_STATIC_BUFFERS;

        let (mut system, buffer) = system_with_thread();
        let tls = system.kernel.current().unwrap().tls;
        let page = tls & !0xFFF;
        let apt = || Target::service("APT:A".into(), 0);
        let word = |value: u32| value.to_le_bytes();

        // where replies put a parameter's buffer.
        system.memory.write_bytes(tls + TLS_IPC_STATIC_BUFFERS, &word(Descriptor::static_buffer(0x100, 0)));
        system.memory.write_bytes(tls + TLS_IPC_STATIC_BUFFERS + 4, &word(page + 0xC00));
        let receive = |system: &mut System| {
            buffer.set(&mut system.memory, 0, Header::new(0x000D, 2, 0).0);
            buffer.set(&mut system.memory, 1, 0x300);
            buffer.set(&mut system.memory, 2, 0x100);
            handle_request(system, Target::service("APT:A".into(), 0));
            (1..=4).chain([6]).map(|i| buffer.get(&mut system.memory, i)).collect::<Vec<_>>()
        };

        // Initialize(application, attributes), then take the wakeup.
        buffer.set(&mut system.memory, 0, Header::new(0x0002, 2, 0).0);
        buffer.set(&mut system.memory, 1, 0x300);
        handle_request(&mut system, apt());
        assert_eq!(receive(&mut system)[..3], [0, 0, 1]);

        // PrepareToStartLibraryApplet(error display)
        buffer.set(&mut system.memory, 0, Header::new(0x0018, 1, 0).0);
        buffer.set(&mut system.memory, 1, 0x406);
        handle_request(&mut system, apt());

        // SendParameter(application, applet, request) with the capture info,
        // whose first word is the size of the capture.
        system.memory.write_bytes(page + 0x800, &word(0x11_8000));
        buffer.set(&mut system.memory, 0, Header::new(0x000C, 4, 4).0);
        for (i, value) in [0x300, 0x406, 2, 0x20, 0, 0, Descriptor::static_buffer(0x20, 0), page + 0x800]
            .into_iter()
            .enumerate()
        {
            buffer.set(&mut system.memory, i as u32 + 1, value);
        }
        handle_request(&mut system, apt());
        // a glance hands out a handle as well, closing it must leave the
        // block to whoever receives the answer.
        buffer.set(&mut system.memory, 0, Header::new(0x000E, 2, 0).0);
        handle_request(&mut system, apt());
        let glanced = buffer.get(&mut system.memory, 6);
        assert!(system.kernel.handles.close(&mut system.kernel.objects, glanced));
        let response = receive(&mut system);
        assert_eq!(response[..3], [0, 0x406, 3], "the applet answers the request");
        let block = system.kernel.resolve(response[4]).and_then(|object| system.kernel.objects.get(object));
        assert!(
            matches!(block, Some(crate::kernel::object::KObject::SharedMemory(_))),
            "with a block for the capture"
        );

        // StartLibraryApplet(applet, size, handle, buffer)
        buffer.set(&mut system.memory, 0, Header::new(0x001E, 2, 4).0);
        for (i, value) in [0x406, 0x40, 0, 0, Descriptor::static_buffer(0x40, 0), page + 0x800]
            .into_iter()
            .enumerate()
        {
            buffer.set(&mut system.memory, i as u32 + 1, value);
        }
        handle_request(&mut system, apt());
        assert_eq!(receive(&mut system)[..4], [0, 0x406, 10, 0x40], "and closes, handing back a result");
        assert!(system.services.unimplemented.is_empty());
    }
}
