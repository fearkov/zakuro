//! cargo run -p zakuro-fs --example romdump -- <rom>

use zakuro_fs::Title;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: romdump <rom.3ds>");
        std::process::exit(2);
    };

    let title = match Title::load(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };

    let h = &title.ncch;
    let x = &title.exheader;
    println!("== NCCH ==");
    println!("  product code   {}", h.product_code);
    println!("  program id     {:016X}", h.program_id);
    println!("  version        {}", h.version);
    println!("  crypto method  0x{:02X} (decrypted: {})", h.crypto_method(), h.is_decrypted());
    println!("  platform       {}", h.platform());
    println!("  exefs          @0x{:X} size 0x{:X}", h.exefs_offset, h.exefs_size);
    println!("  romfs          @0x{:X} size 0x{:X}", h.romfs_offset, h.romfs_size);

    println!("== ExHeader ==");
    println!("  title          {}", x.title);
    println!("  compress code  {}", x.compress_code);
    println!("  .text          0x{:08X} {:>5} pages  size 0x{:X}", x.text.address, x.text.num_pages, x.text.size);
    println!("  .rodata        0x{:08X} {:>5} pages  size 0x{:X}", x.rodata.address, x.rodata.num_pages, x.rodata.size);
    println!("  .data          0x{:08X} {:>5} pages  size 0x{:X}", x.data.address, x.data.num_pages, x.data.size);
    println!("  .bss           0x{:X}", x.bss_size);
    println!("  stack          0x{:X}", x.stack_size);
    println!("  priority       {}", x.main_thread_priority);
    println!("  core / affinity{} / {:#b}", x.ideal_processor, x.affinity_mask);
    println!("  system mode    {:?} ({} MiB app)", x.system_mode, x.system_mode.application_memory() / (1024 * 1024));
    println!("  memory type    {:?}", x.memory_type);
    println!("  handle table   {}", x.handle_table_size);
    println!("  kernel version 0x{:08X}", x.core_version);
    println!("  savedata       0x{:X}", x.savedata_size);
    println!("  services ({}): {}", x.service_access.len(), x.service_access.join(" "));
    println!("  deps ({}): {}", x.dependencies.len(),
        x.dependencies.iter().take(8).map(|d| format!("{d:016X}")).collect::<Vec<_>>().join(" "));

    println!("== ExeFS ==");
    for e in &title.exefs.entries {
        println!("  {:<10} @0x{:08X} size 0x{:X}", e.name, e.offset, e.size);
    }

    match title.code() {
        Ok(code) => {
            let expected = x.data.address + x.data.size - x.text.address;
            println!("== .code ==");
            println!("  decompressed   0x{:X} bytes", code.len());
            println!("  segments span  0x{:X} bytes", expected);
            println!("  first 16 bytes {:02X?}", &code[..16.min(code.len())]);
        }
        Err(e) => println!("== .code == FAILED: {e}"),
    }

    if let Some(fs) = &title.romfs {
        println!("== RomFS ==");
        println!("  level3 base    0x{:X}", fs.base);
        println!("  file data      +0x{:X}", fs.header.file_data_offset);
        let root = fs.root().unwrap();
        println!("  root dirs: {}", fs.subdirs(&root).iter().map(|(_, d)| d.name.clone()).collect::<Vec<_>>().join(" "));
        println!("  root files: {}", fs.files(&root).iter().map(|(_, f)| f.name.clone()).collect::<Vec<_>>().join(" "));
        let (dirs, files) = fs.count_entries();
        println!("  {dirs} directories, {files} files");
    }
}
