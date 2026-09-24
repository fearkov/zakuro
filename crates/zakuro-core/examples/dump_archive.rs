fn main() {
    let image = zakuro_core::services::system_archives::region_manifest();
    let w = |at: usize| u32::from_le_bytes(image[at..at+4].try_into().unwrap());
    println!("total size      0x{:X}", image.len());
    println!("header_size     0x{:X}", w(0x00));
    println!("dir_hash   off 0x{:<6X} size 0x{:X}", w(0x04), w(0x08));
    println!("dir_meta   off 0x{:<6X} size 0x{:X}", w(0x0C), w(0x10));
    println!("file_hash  off 0x{:<6X} size 0x{:X}", w(0x14), w(0x18));
    println!("file_meta  off 0x{:<6X} size 0x{:X}", w(0x1C), w(0x20));
    println!("file_data  off 0x{:X}", w(0x24));
    println!("--- reads the game made ---");
    for off in [0x2C_usize, 0x108] {
        println!("  0x{off:X} -> 0x{:08X}", w(off));
    }
}
