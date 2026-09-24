fn main() {
    let path = std::env::args().nth(1).unwrap();
    let title = zakuro_fs::Title::load(&path).unwrap();
    let code = title.code().unwrap();
    std::fs::write("/tmp/oras_code.bin", &code).unwrap();
    println!("wrote {} bytes; text base 0x{:08X}", code.len(), title.exheader.text.address);
}
