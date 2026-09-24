//! cargo run -p zakuro-core --example watch -- <rom> <addr> [more addrs...]

use zakuro_core::{loader, Config, StepOutcome};
use zakuro_cpu::Bus;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: watch <rom> <addr> [addr...]");
        std::process::exit(2);
    };

    // each argument is either an address or addr:words for a range.
    let mut watches: Vec<u32> = Vec::new();
    for arg in args {
        let (addr, count) = match arg.split_once(':') {
            Some((a, n)) => (a, n.parse::<u32>().unwrap_or(1)),
            None => (arg.as_str(), 1),
        };
        let Ok(base) = u32::from_str_radix(addr.trim_start_matches("0x"), 16) else {
            continue;
        };
        watches.extend((0..count).map(|i| base + i * 4));
    }
    if watches.is_empty() {
        eprintln!("watch: give at least one hex address");
        std::process::exit(2);
    }

    let mut system = loader::load(&path, Config::default()).expect("load");

    let mut previous: Vec<u32> = watches
        .iter()
        .map(|&addr| system.memory.read32(addr))
        .collect();
    println!("watching {} address(es)", watches.len());
    for (addr, value) in watches.iter().zip(&previous) {
        println!("  0x{addr:08X} starts at 0x{value:08X}");
    }

    // enough to get well past initialization, a stalled title will not use it
    // all doing anything new.
    let budget: u64 = 1_500_000_000;
    let mut changes = 0;
    let start = system.cpu.cycles;

    while system.cpu.cycles - start < budget {
        let pc = system.cpu.regs[15];
        let thread = system
            .kernel
            .current()
            .map_or("?".to_owned(), |t| t.name.clone());

        if system.step(None) != StepOutcome::Ran {
            println!("the title stopped");
            break;
        }

        for (index, &addr) in watches.iter().enumerate() {
            let value = system.memory.read32(addr);
            if value != previous[index] {
                println!(
                    "0x{addr:08X}: 0x{:08X} -> 0x{value:08X}  by {thread} at 0x{pc:08X}  (tick {})",
                    previous[index], system.cpu.cycles
                );
                previous[index] = value;
                changes += 1;
                if changes > 200 {
                    println!("(too many changes, stopping)");
                    return;
                }
            }
        }
    }

    println!("finished at tick {}", system.cpu.cycles);
}
