//! fs:USER, the filesystem service, plus the file and directory sessions it
//! hands out.

use std::collections::HashMap;
use std::path::PathBuf;

use zakuro_common::result::{errors, ResultCode};

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::services::host_archive::{self, DirectoryEntry, FormatInfo, HostArchive};
use crate::kernel::object::{ClientSession, KObject};
use crate::services::Target;
use crate::System;

/// archive ids, as FS_ArchiveID defines them.
pub const ARCHIVE_SELF_NCCH: u32 = 0x0000_0003;
pub const ARCHIVE_SAVEDATA: u32 = 0x0000_0004;
pub const ARCHIVE_EXTDATA: u32 = 0x0000_0006;
pub const ARCHIVE_SHARED_EXTDATA: u32 = 0x0000_0007;
pub const ARCHIVE_SYSTEM_SAVEDATA: u32 = 0x0000_0008;
pub const ARCHIVE_SDMC: u32 = 0x0000_0009;
/// "Title access", reaches an NCCH by program ID rather than the implicit self
/// that ARCHIVE_SELF_NCCH gives.
pub const ARCHIVE_NCCH: u32 = 0x2345_678A;

/// media the NCCH archive's path can select.
const MEDIA_NAND: u8 = 0;
const MEDIA_GAMECARD: u8 = 2;

/// high title IDs of the two NAND archives a game reads shared data from.
const SHARED_DATA_ARCHIVE: u32 = 0x0004_009B;
const SYSTEM_DATA_ARCHIVE: u32 = 0x0004_00DB;

/// low title IDs of the individual shared data files.
const SYSTEM_FILE_MII_DATA: u32 = 0x0001_0202;
const SYSTEM_FILE_BAD_WORD_LIST: u32 = 0x0001_0302;
const SYSTEM_FILE_REGION_MANIFEST: u32 = 0x0001_0402;
const SYSTEM_FILE_TLS_ROOT_CERTIFICATES: u32 = 0x0001_0602;
const SYSTEM_FILE_SHARED_FONT: u32 = 0x0001_4002;

/// which title an NCCH-archive path points at.
#[derive(Debug, Clone, Copy, Default)]
pub struct NcchArchivePath {
    pub low_program_id: u32,
    pub high_program_id: u32,
    pub media_type: u8,
}

impl NcchArchivePath {
    /// decodes the 16-byte binary path, the program ID as two words, then
    /// the media type as a single byte at offset 8.
    fn decode(system: &mut System, path_size: u32, path_ptr: u32) -> Option<NcchArchivePath> {
        use zakuro_cpu::Bus;
        if path_size < 12 {
            return None;
        }
        Some(NcchArchivePath {
            low_program_id: system.memory.read32(path_ptr),
            high_program_id: system.memory.read32(path_ptr + 4),
            media_type: system.memory.read8(path_ptr + 8),
        })
    }

    /// a human-readable name for the shared data file this points at, for logs.
    fn system_file_name(&self) -> Option<&'static str> {
        match (self.high_program_id, self.low_program_id) {
            (SHARED_DATA_ARCHIVE, SYSTEM_FILE_MII_DATA) => Some("Mii data"),
            (SHARED_DATA_ARCHIVE, SYSTEM_FILE_REGION_MANIFEST) => Some("region manifest"),
            (SHARED_DATA_ARCHIVE, SYSTEM_FILE_TLS_ROOT_CERTIFICATES) => {
                Some("TLS root certificates")
            }
            (SHARED_DATA_ARCHIVE, SYSTEM_FILE_SHARED_FONT) => Some("shared font"),
            (SYSTEM_DATA_ARCHIVE, SYSTEM_FILE_BAD_WORD_LIST) => Some("bad word list"),
            _ => None,
        }
    }
}

/// path encodings.
pub const PATH_INVALID: u32 = 0;
pub const PATH_EMPTY: u32 = 1;
pub const PATH_BINARY: u32 = 2;
pub const PATH_ASCII: u32 = 3;
pub const PATH_UTF16: u32 = 4;

/// which part of its own NCCH a title is asking for.
const SELF_NCCH_ROMFS: u32 = 0;
const SELF_NCCH_CODE: u32 = 1;
const SELF_NCCH_EXEFS: u32 = 2;

/// OpenFile flags.
const OPEN_CREATE: u32 = 1 << 2;

/// size of one entry a directory read returns.
const DIRECTORY_ENTRY_SIZE: u32 = 0x228;

/// an open file, and where its bytes come from.
#[derive(Debug, Clone)]
pub enum FileBacking {
    /// a window into the loaded ROM image.
    RomImage { offset: u64, size: u64 },
    /// a file in one of the writable archives, kept on the host.
    Host(PathBuf),
    /// read-only bytes generated in memory.
    Memory(Vec<u8>),
}

impl FileBacking {
    pub fn size(&self) -> u64 {
        match self {
            FileBacking::RomImage { size, .. } => *size,
            FileBacking::Host(path) => host_archive::size_of(path),
            FileBacking::Memory(data) => data.len() as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenFile {
    pub path: String,
    pub backing: FileBacking,
}

#[derive(Debug, Clone)]
pub struct OpenArchive {
    pub id: u32,
    /// for [ARCHIVE_NCCH], which title the archive was opened on, a file
    /// opened through it means nothing without knowing that.
    pub ncch_path: Option<NcchArchivePath>,
    /// for the writable archives, the host directory behind it.
    pub host: Option<HostArchive>,
}

/// an open directory, its listing, taken when it was opened, and how much
/// of it has been read.
#[derive(Debug, Clone)]
pub struct OpenDirectory {
    entries: Vec<DirectoryEntry>,
    cursor: usize,
}

pub struct FsState {
    /// where the writable archives live on the host.
    pub user_dir: PathBuf,
    pub archives: HashMap<u64, OpenArchive>,
    pub next_archive: u64,
    pub files: HashMap<u32, OpenFile>,
    pub next_file: u32,
    pub directories: HashMap<u32, OpenDirectory>,
    pub next_directory: u32,
    pub priority: u32,
}

impl Default for FsState {
    fn default() -> Self {
        FsState {
            user_dir: PathBuf::from("user"),
            archives: HashMap::new(),
            next_archive: 0,
            files: HashMap::new(),
            next_file: 0,
            directories: HashMap::new(),
            next_directory: 0,
            priority: 0,
        }
    }
}

impl FsState {
    fn add_archive(&mut self, archive: OpenArchive) -> u64 {
        self.next_archive += 1;
        let handle = self.next_archive;
        self.archives.insert(handle, archive);
        handle
    }

    fn add_file(&mut self, file: OpenFile) -> u32 {
        self.next_file += 1;
        let id = self.next_file;
        self.files.insert(id, file);
        id
    }

    fn add_directory(&mut self, directory: OpenDirectory) -> u32 {
        self.next_directory += 1;
        let id = self.next_directory;
        self.directories.insert(id, directory);
        id
    }
}

pub fn handle(
    system: &mut System,
    buffer: &CommandBuffer,
    header: Header,
    target: &Target,
) -> bool {
    // file and directory sessions carry the object id in the session's
    // subhandle.
    if let Target::Service { name, subhandle } = target {
        match name.as_str() {
            "FSFile" => return file_command(system, buffer, header, *subhandle),
            "FSDirectory" => return directory_command(system, buffer, header, *subhandle),
            _ => {}
        }
    }
    user_command(system, buffer, header)
}

/// replies with nothing but result's code, success or not.
fn reply_result(system: &mut System, buffer: &CommandBuffer, command: u16, result: Result<(), ResultCode>) {
    match result {
        Ok(()) => buffer.reply(&mut system.memory, command, &[]),
        Err(code) => buffer.reply_error(&mut system.memory, command, code.0),
    }
}

/// reads an IPC path argument as raw bytes.
fn path_bytes(system: &mut System, size: u32, pointer: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; size.min(0x400) as usize];
    system.memory.read_bytes(pointer, &mut bytes);
    bytes
}

/// the host directory behind a writable archive, whether or not it has
/// been created yet. None for archives that are not writable.
fn host_archive_for(system: &System, archive_id: u32, path: &[u8]) -> Option<HostArchive> {
    let base = &system.services.fs.user_dir;
    let word = |i: usize| {
        path.get(i..i + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    match archive_id {
        ARCHIVE_SAVEDATA => Some(HostArchive::save_data(base, system.kernel.program_id)),
        // path, media type, then the 64-bit extra data id.
        ARCHIVE_EXTDATA | ARCHIVE_SHARED_EXTDATA => {
            let id = word(4) as u64 | ((word(8) as u64) << 32);
            Some(HostArchive::ext_data(base, id, archive_id == ARCHIVE_SHARED_EXTDATA))
        }
        ARCHIVE_SYSTEM_SAVEDATA => {
            let id = word(4) as u64 | ((word(0) as u64) << 32);
            Some(HostArchive::system_save_data(base, id))
        }
        ARCHIVE_SDMC => Some(HostArchive::sdmc(base)),
        _ => None,
    }
}

/// opens a writable archive, failing the way a console does when it was
/// never created, each kind has its own code, and titles tell "create it"
/// apart from "it is broken" by exactly which one comes back.
fn open_host_archive(system: &System, archive_id: u32, path: &[u8]) -> Result<HostArchive, ResultCode> {
    let archive = host_archive_for(system, archive_id, path).ok_or(errors::FS_NOT_FOUND)?;
    if archive_id == ARCHIVE_SDMC {
        // the SD card is simply there.
        let _ = std::fs::create_dir_all(archive.resolve("").unwrap_or_default());
        return Ok(archive);
    }
    if archive.exists() {
        return Ok(archive);
    }
    Err(match archive_id {
        ARCHIVE_EXTDATA => errors::FS_NOT_FOUND_INVALID_STATE,
        _ => errors::FS_NOT_FORMATTED,
    })
}

/// the writable archive an archive handle refers to.
fn archive_of(system: &System, handle: u64) -> Result<HostArchive, ResultCode> {
    match system.services.fs.archives.get(&handle) {
        Some(OpenArchive { host: Some(host), .. }) => Ok(host.clone()),
        // read-only archives have nowhere to put anything.
        Some(_) => Err(errors::FS_UNEXPECTED_FILE_OR_DIRECTORY),
        None => Err(errors::FS_ARCHIVE_NOT_MOUNTED),
    }
}

fn user_command(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // initialize / InitializeWithSdkVersion
        0x0801 | 0x0861 => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }

        // OpenFile(transaction, archive handle, path type, path size, flags,
        //          attributes, <path>)
        0x0802 => {
            let archive = read_u64(system, buffer, 2);
            let path_type = buffer.get(&mut system.memory, 4);
            let path_size = buffer.get(&mut system.memory, 5);
            let flags = buffer.get(&mut system.memory, 6);
            let path_ptr = buffer.get(&mut system.memory, 9);
            let open = system.services.fs.archives.get(&archive).cloned();

            let result = match open {
                Some(OpenArchive { host: Some(host), .. }) => {
                    open_host_file(system, &host, path_type, path_size, path_ptr, flags)
                }
                Some(open) => {
                    open_file(system, open.id, open.ncch_path, path_type, path_size, path_ptr)
                        .ok_or(errors::FS_NOT_FOUND)
                }
                None => Err(errors::FS_ARCHIVE_NOT_MOUNTED),
            };
            reply_with_file(system, buffer, command, result);
            true
        }

        // OpenFileDirectly(transaction, archive id, archive path type,
        //   archive path size, file path type, file path size, flags,
        //   attributes, <archive path>, <file path>)
        0x0803 => {
            let archive_id = buffer.get(&mut system.memory, 2);
            let archive_path_size = buffer.get(&mut system.memory, 4);
            let file_path_type = buffer.get(&mut system.memory, 5);
            let file_path_size = buffer.get(&mut system.memory, 6);
            let flags = buffer.get(&mut system.memory, 7);
            let archive_path_ptr = buffer.get(&mut system.memory, 10);
            let file_path_ptr = buffer.get(&mut system.memory, 12);

            let archive_path = path_bytes(system, archive_path_size, archive_path_ptr);
            let result = if host_archive_for(system, archive_id, &archive_path).is_some() {
                open_host_archive(system, archive_id, &archive_path).and_then(|host| {
                    open_host_file(system, &host, file_path_type, file_path_size, file_path_ptr, flags)
                })
            } else {
                let ncch_path = if archive_id == ARCHIVE_NCCH {
                    NcchArchivePath::decode(system, archive_path_size, archive_path_ptr)
                } else {
                    None
                };
                open_file(system, archive_id, ncch_path, file_path_type, file_path_size, file_path_ptr)
                    .ok_or(errors::FS_NOT_FOUND)
            };
            reply_with_file(system, buffer, command, result);
            true
        }

        // DeleteFile / DeleteDirectory / DeleteDirectoryRecursively
        //   (transaction, archive handle, path type, path size, <path>)
        0x0804 | 0x0806 | 0x0807 => {
            let archive = read_u64(system, buffer, 2);
            let path_type = buffer.get(&mut system.memory, 4);
            let path_size = buffer.get(&mut system.memory, 5);
            let path_ptr = buffer.get(&mut system.memory, 7);
            let path = decode_path(system, path_type, path_size, path_ptr);
            let result = archive_of(system, archive).and_then(|host| match command {
                0x0804 => host.delete_file(&path),
                _ => host.delete_directory(&path, command == 0x0807),
            });
            log::debug!("fs: delete '{path}' (command 0x{command:04X}) -> {result:?}");
            reply_result(system, buffer, command, result);
            true
        }

        // RenameFile / RenameDirectory(transaction, source archive, source
        //   path type, source path size, destination archive, destination
        //   path type, destination path size, <source>, <destination>)
        0x0805 | 0x080A => {
            let source_archive = read_u64(system, buffer, 2);
            let source_type = buffer.get(&mut system.memory, 4);
            let source_size = buffer.get(&mut system.memory, 5);
            let dest_type = buffer.get(&mut system.memory, 8);
            let dest_size = buffer.get(&mut system.memory, 9);
            let source_ptr = buffer.get(&mut system.memory, 11);
            let dest_ptr = buffer.get(&mut system.memory, 13);
            let source = decode_path(system, source_type, source_size, source_ptr);
            let dest = decode_path(system, dest_type, dest_size, dest_ptr);
            // renames only ever happen within one archive.
            let result = archive_of(system, source_archive).and_then(|host| {
                if command == 0x0805 {
                    host.rename_file(&source, &dest)
                } else {
                    host.rename_directory(&source, &dest)
                }
            });
            log::debug!("fs: rename '{source}' -> '{dest}' -> {result:?}");
            reply_result(system, buffer, command, result);
            true
        }

        // CreateFile(transaction, archive handle, path type, path size,
        //   attributes, size u64, <path>)
        0x0808 => {
            let archive = read_u64(system, buffer, 2);
            let path_type = buffer.get(&mut system.memory, 4);
            let path_size = buffer.get(&mut system.memory, 5);
            let size = read_u64(system, buffer, 7);
            let path_ptr = buffer.get(&mut system.memory, 10);
            let path = decode_path(system, path_type, path_size, path_ptr);
            let result = archive_of(system, archive).and_then(|host| host.create_file(&path, size));
            log::debug!("fs: create file '{path}' of 0x{size:X} bytes -> {result:?}");
            reply_result(system, buffer, command, result);
            true
        }

        // CreateDirectory(transaction, archive handle, path type, path
        //   size, attributes, <path>)
        0x0809 => {
            let archive = read_u64(system, buffer, 2);
            let path_type = buffer.get(&mut system.memory, 4);
            let path_size = buffer.get(&mut system.memory, 5);
            let path_ptr = buffer.get(&mut system.memory, 8);
            let path = decode_path(system, path_type, path_size, path_ptr);
            let result = archive_of(system, archive).and_then(|host| host.create_directory(&path));
            log::debug!("fs: create directory '{path}' -> {result:?}");
            reply_result(system, buffer, command, result);
            true
        }

        // OpenDirectory(archive handle, path type, path size, <path>)
        0x080B => {
            let archive = read_u64(system, buffer, 1);
            let path_type = buffer.get(&mut system.memory, 3);
            let path_size = buffer.get(&mut system.memory, 4);
            let path_ptr = buffer.get(&mut system.memory, 6);
            let path = decode_path(system, path_type, path_size, path_ptr);
            match archive_of(system, archive).and_then(|host| host.list(&path)) {
                Ok(entries) => {
                    log::debug!("fs: opened directory '{path}' ({} entries)", entries.len());
                    let id = system
                        .services
                        .fs
                        .add_directory(OpenDirectory { entries, cursor: 0 });
                    let handle = make_session(system, "FSDirectory", id);
                    buffer.reply_with_handle(&mut system.memory, command, handle);
                }
                Err(code) => {
                    log::debug!("fs: directory '{path}' -> {code:?}");
                    buffer.reply_error(&mut system.memory, command, code.0);
                }
            }
            true
        }

        // OpenArchive(archive id, path type, path size, <path>)
        0x080C => {
            let archive_id = buffer.get(&mut system.memory, 1);
            let path_size = buffer.get(&mut system.memory, 3);
            let path_ptr = buffer.get(&mut system.memory, 5);
            let path = path_bytes(system, path_size, path_ptr);

            let host = if host_archive_for(system, archive_id, &path).is_some() {
                match open_host_archive(system, archive_id, &path) {
                    Ok(host) => Some(host),
                    Err(code) => {
                        log::debug!("fs: archive 0x{archive_id:08X} {path:02X?} -> {code:?}");
                        buffer.reply_error(&mut system.memory, command, code.0);
                        return true;
                    }
                }
            } else {
                None
            };
            let ncch_path = if archive_id == ARCHIVE_NCCH {
                NcchArchivePath::decode(system, path_size, path_ptr)
            } else {
                None
            };
            let handle = system.services.fs.add_archive(OpenArchive {
                id: archive_id,
                ncch_path,
                host,
            });
            log::debug!("fs: opened archive 0x{archive_id:08X} -> {handle}");
            buffer.set(&mut system.memory, 0, Header::new(command, 3, 0).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, handle as u32);
            buffer.set(&mut system.memory, 3, (handle >> 32) as u32);
            true
        }

        // ControlArchive(archive handle, action, input size, output size,
        //   <input>, <output>). The one action titles use commits save
        //   data, which host files do on every write anyway.
        0x080D => {
            let descriptors: Vec<u32> = (6..10).map(|i| buffer.get(&mut system.memory, i)).collect();
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 4).0);
            buffer.set(&mut system.memory, 1, 0);
            for (i, word) in descriptors.into_iter().enumerate() {
                buffer.set(&mut system.memory, 2 + i as u32, word);
            }
            true
        }

        // CloseArchive
        0x080E => {
            let archive = read_u64(system, buffer, 1);
            system.services.fs.archives.remove(&archive);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }

        // FormatThisUserSaveData(block size, directories, files, directory
        //   buckets, file buckets, duplicate data)
        0x080F => {
            let info = FormatInfo {
                total_size: buffer.get(&mut system.memory, 1).wrapping_mul(512),
                directories: buffer.get(&mut system.memory, 2),
                files: buffer.get(&mut system.memory, 3),
                duplicate_data: buffer.get(&mut system.memory, 6) & 0xFF != 0,
            };
            let archive = HostArchive::save_data(&system.services.fs.user_dir, system.kernel.program_id);
            let result = archive.format(info);
            reply_result(system, buffer, command, result);
            true
        }

        // FormatSaveData(archive id, path type, path size, block size,
        //   directories, files, directory buckets, file buckets, duplicate
        //   data, <path>)
        0x084C => {
            let archive_id = buffer.get(&mut system.memory, 1);
            let path_size = buffer.get(&mut system.memory, 3);
            let info = FormatInfo {
                total_size: buffer.get(&mut system.memory, 4).wrapping_mul(512),
                directories: buffer.get(&mut system.memory, 5),
                files: buffer.get(&mut system.memory, 6),
                duplicate_data: buffer.get(&mut system.memory, 9) & 0xFF != 0,
            };
            let path_ptr = buffer.get(&mut system.memory, 11);
            let path = path_bytes(system, path_size, path_ptr);
            let result = match host_archive_for(system, archive_id, &path) {
                Some(archive) if archive_id == ARCHIVE_SAVEDATA => archive.format(info),
                _ => Err(errors::FS_INVALID_PATH),
            };
            log::debug!("fs: format archive 0x{archive_id:08X} with {info:?} -> {result:?}");
            reply_result(system, buffer, command, result);
            true
        }

        // GetFreeBytes(archive handle) -> u64. Plenty.
        0x0812 => {
            let free: u64 = 512 * 1024 * 1024;
            buffer.reply(&mut system.memory, command, &[free as u32, (free >> 32) as u32]);
            true
        }

        // GetCardType, a CTR card.
        0x0813 => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }

        // GetSdmcArchiveResource / GetNandArchiveResource /
        // GetArchiveResource(media type), sector size, cluster size, total
        // and free clusters.
        0x0814 | 0x0815 | 0x0849 => {
            buffer.reply(&mut system.memory, command, &[0x200, 0x4000, 0x8_0000, 0x8_0000]);
            true
        }

        // IsSdmcDetected / IsSdmcWritable / CardSlotIsInserted
        0x0817 | 0x0818 | 0x0821 => {
            buffer.reply(&mut system.memory, command, &[1]);
            true
        }

        // CheckAuthorityToAccessExtSaveData, yes.
        0x083D => {
            buffer.reply(&mut system.memory, command, &[1]);
            true
        }

        // AbnegateAccessRight / SetArchivePriority / SetFsCompatibilityInfo
        0x0840 | 0x085A | 0x085D => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }

        // GetFormatInfo(archive id, path type, path size, <path>) -> total
        // size, directories, files, duplicate data
        0x0845 => {
            let archive_id = buffer.get(&mut system.memory, 1);
            let path_size = buffer.get(&mut system.memory, 3);
            let path_ptr = buffer.get(&mut system.memory, 5);
            let path = path_bytes(system, path_size, path_ptr);
            match host_archive_for(system, archive_id, &path).and_then(|a| a.format_info()) {
                Some(info) => buffer.reply(
                    &mut system.memory,
                    command,
                    &[info.total_size, info.directories, info.files, info.duplicate_data as u32],
                ),
                None => {
                    log::debug!("fs: format info of archive 0x{archive_id:08X}: not formatted");
                    buffer.reply_error(&mut system.memory, command, errors::FS_NOT_FORMATTED.0);
                }
            }
            true
        }

        // CreateExtSaveData(media type, id low, id high, unknown,
        //   directories, files, size limit u64, icon size, <icon>)
        0x0851 => {
            let path: Vec<u8> = (1..4)
                .flat_map(|i| buffer.get(&mut system.memory, i).to_le_bytes())
                .collect();
            let info = FormatInfo {
                total_size: 0,
                directories: buffer.get(&mut system.memory, 5),
                files: buffer.get(&mut system.memory, 6),
                duplicate_data: false,
            };
            let icon_descriptor = buffer.get(&mut system.memory, 10);
            let icon_ptr = buffer.get(&mut system.memory, 11);
            let result = match host_archive_for(system, ARCHIVE_EXTDATA, &path) {
                // creating extra data that is already there leaves its
                // contents alone.
                Some(archive) if archive.exists() => Ok(()),
                Some(archive) => archive.format(info),
                None => Err(errors::FS_INVALID_PATH),
            };
            log::debug!("fs: create extra data {path:02X?} -> {result:?}");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, result.err().map_or(0, |code| code.0));
            buffer.set(&mut system.memory, 2, icon_descriptor);
            buffer.set(&mut system.memory, 3, icon_ptr);
            true
        }

        // DeleteExtSaveData(media type, id low, id high, unknown)
        0x0852 => {
            let path: Vec<u8> = (1..4)
                .flat_map(|i| buffer.get(&mut system.memory, i).to_le_bytes())
                .collect();
            let result = host_archive_for(system, ARCHIVE_EXTDATA, &path)
                .ok_or(errors::FS_INVALID_PATH)
                .and_then(|archive| archive.delete());
            reply_result(system, buffer, command, result);
            true
        }

        // SetPriority / GetPriority
        0x0862 => {
            system.services.fs.priority = buffer.get(&mut system.memory, 1);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        0x0863 => {
            let priority = system.services.fs.priority;
            buffer.reply(&mut system.memory, command, &[priority]);
            true
        }

        _ => false,
    }
}

/// opens a file inside a writable archive.
fn open_host_file(
    system: &mut System,
    host: &HostArchive,
    path_type: u32,
    path_size: u32,
    path_ptr: u32,
    flags: u32,
) -> Result<u32, ResultCode> {
    let path = decode_path(system, path_type, path_size, path_ptr);
    let result = host.open_file(&path, flags & OPEN_CREATE != 0);
    log::debug!("fs: open '{path}' (flags {flags:#X}) -> {:?}", result.as_ref().map(|_| ()));
    let host_path = result?;
    Ok(system.services.fs.add_file(OpenFile {
        path,
        backing: FileBacking::Host(host_path),
    }))
}

fn reply_with_file(
    system: &mut System,
    buffer: &CommandBuffer,
    command: u16,
    result: Result<u32, ResultCode>,
) {
    match result {
        Ok(id) => {
            let handle = make_session(system, "FSFile", id);
            buffer.reply_with_handle(&mut system.memory, command, handle);
        }
        Err(code) => buffer.reply_error(&mut system.memory, command, code.0),
    }
}

/// commands sent to a file session.
fn file_command(
    system: &mut System,
    buffer: &CommandBuffer,
    header: Header,
    file_id: u32,
) -> bool {
    let command = header.command_id();
    if log::log_enabled!(log::Level::Trace) {
        let name = system
            .services
            .fs
            .files
            .get(&file_id)
            .map(|f| f.path.clone())
            .unwrap_or_else(|| "<closed>".into());
        log::trace!("fs: file {file_id} ('{name}') command 0x{command:04X}");
    }
    let backing = system.services.fs.files.get(&file_id).map(|f| f.backing.clone());
    match command {
        // read(offset u64, size, <mapped buffer>)
        0x0802 => {
            let offset = read_u64(system, buffer, 1);
            let size = buffer.get(&mut system.memory, 3);
            let dest = buffer.get(&mut system.memory, 5);
            log::trace!("fs: file {file_id} read offset 0x{offset:X} size 0x{size:X}");

            let read = read_file(system, file_id, offset, size, dest);
            buffer.set(&mut system.memory, 0, Header::new(command, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, read);
            buffer.set(&mut system.memory, 3, (size << 4) | 0xC);
            buffer.set(&mut system.memory, 4, dest);
            true
        }
        // write(offset u64, size, flags, <mapped buffer>)
        0x0803 => {
            let offset = read_u64(system, buffer, 1);
            let size = buffer.get(&mut system.memory, 3);
            let source = buffer.get(&mut system.memory, 6);

            let written = match backing {
                Some(FileBacking::Host(path)) => {
                    let mut data = vec![0u8; size as usize];
                    system.memory.read_bytes(source, &mut data);
                    host_archive::write_at(&path, offset, &data) as u32
                }
                // anything else is read-only, the ROM image and the
                // generated system archives have nowhere to put writes.
                _ => 0,
            };

            buffer.set(&mut system.memory, 0, Header::new(command, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, written);
            buffer.set(&mut system.memory, 3, (size << 4) | 0xA);
            buffer.set(&mut system.memory, 4, source);
            true
        }
        // GetSize -> u64
        0x0804 => {
            let size = backing.map_or(0, |b| b.size());
            buffer.reply(
                &mut system.memory,
                command,
                &[size as u32, (size >> 32) as u32],
            );
            true
        }
        // SetSize(size u64)
        0x0805 => {
            let size = read_u64(system, buffer, 1);
            let result = match backing {
                Some(FileBacking::Host(path)) => host_archive::set_size(&path, size),
                _ => Ok(()),
            };
            reply_result(system, buffer, command, result);
            true
        }
        // GetAttributes, an ordinary file.
        0x0806 => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // close
        0x0808 => {
            system.services.fs.files.remove(&file_id);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetAttributes / Flush / SetPriority, host writes are already durable.
        0x0807 | 0x0809 | 0x080A => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetPriority
        0x080B => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // OpenLinkFile, hand back another session onto the same file.
        0x080C => {
            let handle = make_session(system, "FSFile", file_id);
            buffer.reply_with_handle(&mut system.memory, command, handle);
            true
        }
        _ => false,
    }
}

/// commands sent to a directory session.
fn directory_command(
    system: &mut System,
    buffer: &CommandBuffer,
    header: Header,
    directory_id: u32,
) -> bool {
    let command = header.command_id();
    match command {
        // read(count, <mapped buffer>) -> entries read
        0x0801 => {
            let count = buffer.get(&mut system.memory, 1);
            let descriptor = buffer.get(&mut system.memory, 2);
            let dest = buffer.get(&mut system.memory, 3);
            let entries: Vec<DirectoryEntry> = match system.services.fs.directories.get_mut(&directory_id) {
                Some(directory) => {
                    let end = (directory.cursor + count as usize).min(directory.entries.len());
                    let batch = directory.entries[directory.cursor..end].to_vec();
                    directory.cursor = end;
                    batch
                }
                None => Vec::new(),
            };
            for (i, entry) in entries.iter().enumerate() {
                let raw = encode_directory_entry(entry);
                system
                    .memory
                    .write_bytes(dest + i as u32 * DIRECTORY_ENTRY_SIZE, &raw);
            }
            buffer.set(&mut system.memory, 0, Header::new(command, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, entries.len() as u32);
            buffer.set(&mut system.memory, 3, descriptor);
            buffer.set(&mut system.memory, 4, dest);
            true
        }
        // close
        0x0802 => {
            system.services.fs.directories.remove(&directory_id);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        _ => false,
    }
}

/// lays out one directory entry the way FSDir:Read returns them, the
/// UTF-16 name, an 8.3 short name and extension, flags, and the size.
fn encode_directory_entry(entry: &DirectoryEntry) -> [u8; DIRECTORY_ENTRY_SIZE as usize] {
    let mut raw = [0u8; DIRECTORY_ENTRY_SIZE as usize];
    for (i, unit) in entry.name.encode_utf16().take(0x105).enumerate() {
        raw[i * 2..i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
    }
    let (stem, extension) = match entry.name.rsplit_once('.') {
        Some((stem, extension)) if !entry.is_directory => (stem, extension),
        _ => (entry.name.as_str(), ""),
    };
    for (i, byte) in stem.bytes().take(8).enumerate() {
        raw[0x20C + i] = byte.to_ascii_uppercase();
    }
    for (i, byte) in extension.bytes().take(3).enumerate() {
        raw[0x216 + i] = byte.to_ascii_uppercase();
    }
    raw[0x21A] = 1;
    raw[0x21C] = entry.is_directory as u8;
    raw[0x21E] = !entry.is_directory as u8;
    raw[0x220..0x228].copy_from_slice(&entry.size.to_le_bytes());
    raw
}

fn read_u64(system: &mut System, buffer: &CommandBuffer, index: u32) -> u64 {
    let low = buffer.get(&mut system.memory, index) as u64;
    let high = buffer.get(&mut system.memory, index + 1) as u64;
    low | (high << 32)
}

/// creates the session object a file or directory is reached through.
fn make_session(system: &mut System, service: &str, id: u32) -> u32 {
    let object = system
        .kernel
        .objects
        .insert(KObject::ClientSession(ClientSession {
            service: service.into(),
            subhandle: id,
        }));
    system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, service)
}

/// opens a file in one of the read-only archives, the title's own NCCH
/// and the system's shared data.
fn open_file(
    system: &mut System,
    archive_id: u32,
    archive_path: Option<NcchArchivePath>,
    path_type: u32,
    path_size: u32,
    path_ptr: u32,
) -> Option<u32> {
    match archive_id {
        ARCHIVE_SELF_NCCH => open_self_ncch(system, path_type, path_size, path_ptr),
        ARCHIVE_NCCH if path_type == PATH_BINARY && path_size >= 12 => {
            use zakuro_cpu::Bus;
            let access_type = system.memory.read32(path_ptr);
            if access_type != 0 {
                log::warn!("fs: NCCH archive access type {access_type} is not implemented");
                return None;
            }

            // the media type in the *archive* path decides what this is, a game
            // reaching into its own cartridge, or, far more often, a read of
            // one of the system's shared data archives on NAND, which has
            // nothing to do with this title's content.
            let archive_path = archive_path.unwrap_or_default();
            match archive_path.media_type {
                MEDIA_NAND => open_system_archive(system, archive_path),
                MEDIA_GAMECARD => {
                    let kind = system.memory.read32(path_ptr + 8);
                    open_ncch_kind(system, kind)
                }
                other => {
                    log::warn!("fs: NCCH archive media type {other} is not implemented");
                    None
                }
            }
        }
        other => {
            log::warn!("fs: unknown archive 0x{other:08X}");
            None
        }
    }
}

fn open_self_ncch(
    system: &mut System,
    path_type: u32,
    path_size: u32,
    path_ptr: u32,
) -> Option<u32> {
    // a binary path selects which part of the NCCH, an empty path means RomFS.
    let kind = if path_type == PATH_BINARY && path_size >= 4 {
        use zakuro_cpu::Bus;
        system.memory.read32(path_ptr)
    } else {
        SELF_NCCH_ROMFS
    };

    open_ncch_kind(system, kind)
}

/// opens one of the system's shared data archives on NAND.
fn open_system_archive(system: &mut System, path: NcchArchivePath) -> Option<u32> {
    let Some(name) = path.system_file_name() else {
        log::warn!(
            "fs: unknown NAND shared archive {:08X}/{:08X}",
            path.high_program_id,
            path.low_program_id
        );
        return None;
    };

    let data = system_archive_data(path);
    log::debug!("fs: opened the system's {name} ({} bytes)", data.len());
    Some(system.services.fs.add_file(OpenFile {
        path: format!("nand:/{name}"),
        backing: FileBacking::Memory(data),
    }))
}

/// stand-in contents for a shared data archive.
fn system_archive_data(path: NcchArchivePath) -> Vec<u8> {
    use crate::services::system_archives;
    match (path.high_program_id, path.low_program_id) {
        (SHARED_DATA_ARCHIVE, SYSTEM_FILE_REGION_MANIFEST) => system_archives::region_manifest(),
        (SYSTEM_DATA_ARCHIVE, SYSTEM_FILE_BAD_WORD_LIST) => system_archives::bad_word_list(),
        _ => Vec::new(),
    }
}

fn open_ncch_kind(system: &mut System, kind: u32) -> Option<u32> {
    let title = system.title.as_ref()?;
    match kind {
        SELF_NCCH_ROMFS => {
            let romfs = title.romfs.as_ref()?;
            let size = title.ncch.romfs_size.saturating_sub(0x1000);
            let backing = FileBacking::RomImage {
                offset: romfs.base,
                size,
            };
            log::debug!(
                "fs: opened SelfNCCH RomFS at file offset 0x{:X}, {} MiB",
                romfs.base,
                size / (1024 * 1024)
            );
            Some(system.services.fs.add_file(OpenFile {
                path: "romfs:/".into(),
                backing,
            }))
        }
        SELF_NCCH_CODE | SELF_NCCH_EXEFS => {
            let data = title.exefs_file(".code")?.to_vec();
            Some(system.services.fs.add_file(OpenFile {
                path: "exefs:/.code".into(),
                backing: FileBacking::Memory(data),
            }))
        }
        other => {
            log::warn!("fs: SelfNCCH path type {other} is not implemented");
            None
        }
    }
}

fn read_file(system: &mut System, file_id: u32, offset: u64, size: u32, dest: u32) -> u32 {
    let Some(file) = system.services.fs.files.get(&file_id).cloned() else {
        return 0;
    };

    match file.backing {
        FileBacking::RomImage {
            offset: base,
            size: total,
        } => {
            if offset >= total {
                return 0;
            }
            let count = (size as u64).min(total - offset) as usize;
            let Some(title) = system.title.as_ref() else {
                return 0;
            };
            let start = (base + offset) as usize;
            let Some(slice) = title.image().data().get(start..start + count) else {
                log::warn!("fs: read past the end of the ROM image");
                return 0;
            };
            // copying through a temporary keeps the borrow of title from
            // overlapping the mutable borrow of memory.
            let data = slice.to_vec();
            system.memory.write_bytes(dest, &data);
            count as u32
        }
        FileBacking::Host(path) => {
            let mut data = vec![0u8; size as usize];
            let count = host_archive::read_at(&path, offset, &mut data);
            system.memory.write_bytes(dest, &data[..count]);
            count as u32
        }
        FileBacking::Memory(data) => {
            if offset >= data.len() as u64 {
                return 0;
            }
            let start = offset as usize;
            let count = (size as usize).min(data.len() - start);
            system.memory.write_bytes(dest, &data[start..start + count]);
            count as u32
        }
    }
}

fn decode_path(system: &mut System, path_type: u32, path_size: u32, path_ptr: u32) -> String {
    match path_type {
        PATH_ASCII => system.memory.read_cstring(path_ptr, path_size as usize),
        PATH_UTF16 => {
            let mut bytes = vec![0u8; path_size as usize];
            system.memory.read_bytes(path_ptr, &mut bytes);
            let units: Vec<u16> = bytes
                .as_chunks::<2>().0.iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .take_while(|&u| u != 0)
                .collect();
            String::from_utf16_lossy(&units)
        }
        PATH_EMPTY => String::new(),
        PATH_BINARY => {
            let mut bytes = vec![0u8; path_size.min(64) as usize];
            system.memory.read_bytes(path_ptr, &mut bytes);
            format!("<binary {bytes:02X?}>")
        }
        _ => "<invalid>".into(),
    }
}

/// kept so the descriptor helper stays in scope for the mapped-buffer replies
/// above, which build their descriptors by hand.
pub const _MAPPED_BUFFER_RW: u32 = Descriptor::static_buffer(0, 0);
