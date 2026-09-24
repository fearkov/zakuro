fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    let path = std::env::args().nth(1).unwrap();
    let system = zakuro_core::loader::load(&path, zakuro_core::Config::default()).expect("load");
    println!("--- mappings ---");
    for m in system.memory.mappings() {
        println!("  0x{:08X}..0x{:08X} pa 0x{:08X} {:?} {:?}", m.base, m.base+m.size, m.paddr, m.permission, m.state);
    }
    println!("readable at 0x100000: {}", system.memory.is_readable(0x100000, 4));
}
