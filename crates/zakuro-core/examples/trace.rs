//! cargo run -p zakuro-core --example trace -- <rom> [max-instructions]

use std::collections::VecDeque;

use zakuro_core::{loader, Config, StepOutcome};

const HISTORY: usize = 48;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: trace <rom> [max-instructions]");
        std::process::exit(2);
    };
    let max: u64 = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(50_000_000);

    let mut system = loader::load(&path, Config::default()).expect("load");

    let mut history: VecDeque<(u32, bool)> = VecDeque::with_capacity(HISTORY);
    let faults = system.memory.fault_summary().len();

    for _ in 0..max {
        let pc = system.cpu.regs[15];
        let thumb = system.cpu.cpsr.thumb;
        if history.len() == HISTORY {
            history.pop_front();
        }
        history.push_back((pc, thumb));

        if system.step(None) != StepOutcome::Ran {
            println!("the title stopped");
            dump(&history, &system);
            return;
        }

        // executing from an unmapped page means a jump through a pointer
        // that was never filled in, which is far more informative than the
        // data faults that follow it.
        let pc = system.cpu.regs[15];
        if system.memory.mapping_at(pc).is_none() {
            println!("execution left mapped memory at 0x{pc:08X}");
            dump(&history, &system);
            return;
        }

        if system.broke || !system.fatal_errors.is_empty() {
            println!("stopped: {}", system.fatal_errors.join("; "));
            dump(&history, &system);
            return;
        }

        let now = system.memory.fault_summary().len();
        if now != faults {
            let newest = system
                .memory
                .fault_summary()
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(addr, _)| addr)
                .unwrap_or(0);
            println!(
                "first unmapped access after {} instructions (page 0x{newest:08X})",
                system.cpu.cycles
            );
            dump(&history, &system);
            return;
        }
    }

    println!("ran {} instructions with no unmapped access", system.cpu.cycles);
    println!("{:?}", system.cpu);
}

fn dump(history: &VecDeque<(u32, bool)>, system: &zakuro_core::System) {
    println!("--- last {} instructions ---", history.len());
    for (pc, thumb) in history {
        println!("  0x{pc:08X} {}", if *thumb { "T" } else { "A" });
    }
    println!("{:?}", system.cpu);
    println!("--- mappings ---");
    for mapping in system.memory.mappings() {
        println!(
            "  0x{:08X}..0x{:08X} {:?} {:?}",
            mapping.base,
            mapping.base + mapping.size,
            mapping.permission,
            mapping.state
        );
    }
}
