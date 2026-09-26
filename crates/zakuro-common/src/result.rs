//! 3DS result codes.
//!
//! a result is a packed 32-bit value, with the level in bits 27-31, the
//! summary in 21-26, the module in 10-17 and the description in 0-9.
//!
//! Userland checks the sign bit for failure, but games also compare against
//! exact values, so the individual fields have to be right.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResultCode(pub u32);

/// result of an HLE operation, either a value or a 3DS error code.
pub type ResultValue<T> = Result<T, ResultCode>;

pub const RESULT_SUCCESS: ResultCode = ResultCode(0);

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Success = 0,
    Info = 1,
    Status = 25,
    Temporary = 26,
    Permanent = 27,
    Usage = 28,
    Reinitialize = 29,
    Reset = 30,
    Fatal = 31,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Summary {
    Success = 0,
    NothingHappened = 1,
    WouldBlock = 2,
    OutOfResource = 3,
    NotFound = 4,
    InvalidState = 5,
    NotSupported = 6,
    InvalidArgument = 7,
    WrongArgument = 8,
    Canceled = 9,
    StatusChanged = 10,
    Internal = 11,
    InvalidResultValue = 63,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Module {
    Common = 0,
    Kernel = 1,
    Util = 2,
    FileServer = 3,
    LoaderServer = 4,
    Tcb = 5,
    Os = 6,
    Dbg = 7,
    Dmnt = 8,
    Pdn = 9,
    Gsp = 10,
    I2c = 11,
    Gpio = 12,
    Dd = 13,
    Codec = 14,
    Spi = 15,
    Pxi = 16,
    Fs = 17,
    Di = 18,
    Hid = 19,
    Cam = 20,
    Pi = 21,
    Pm = 22,
    PmLow = 23,
    Fsi = 24,
    Srv = 25,
    Ndm = 26,
    Nwm = 27,
    Soc = 28,
    Ldr = 29,
    Acc = 30,
    RomFs = 31,
    Am = 32,
    Hio = 33,
    Updater = 34,
    Mic = 35,
    Fnd = 36,
    Mp = 37,
    Mpwl = 38,
    Ac = 39,
    Http = 40,
    Dsp = 41,
    Snd = 42,
    Dlp = 43,
    HioLow = 44,
    Csnd = 45,
    Ssl = 46,
    AmLow = 47,
    Nex = 48,
    Friends = 49,
    Rdt = 50,
    Applet = 51,
    Nim = 52,
    Ptm = 53,
    Midi = 54,
    Mc = 55,
    Swc = 56,
    FatFs = 57,
    Ngc = 58,
    Card = 59,
    CardNor = 60,
    Sdmc = 61,
    Boss = 62,
    Dbm = 63,
    Config = 64,
    Ps = 65,
    Cec = 66,
    Ir = 67,
    Uds = 68,
    Pl = 69,
    Cup = 70,
    Gyroscope = 71,
    Mcu = 72,
    Ns = 73,
    News = 74,
    Ro = 75,
    Gd = 76,
    CardSpi = 77,
    Ec = 78,
    WebBrowser = 79,
    Test = 80,
    Enc = 81,
    Pia = 82,
    Act = 83,
    Vctl = 84,
    Olv = 85,
    Neia = 86,
    Npns = 87,
    Avd = 90,
    L2b = 91,
    Mvd = 92,
    Nfc = 93,
    Uart = 94,
    Spm = 95,
    Qtm = 96,
    Nfp = 97,
    Application = 254,
    InvalidResult = 255,
}

impl ResultCode {
    pub const fn new(level: Level, summary: Summary, module: Module, description: u32) -> Self {
        ResultCode(
            (description & 0x3FF)
                | ((module as u32 & 0xFF) << 10)
                | ((summary as u32 & 0x3F) << 21)
                | ((level as u32 & 0x1F) << 27),
        )
    }

    #[inline]
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_error(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn description(self) -> u32 {
        self.0 & 0x3FF
    }

    #[inline]
    pub const fn module(self) -> u32 {
        (self.0 >> 10) & 0xFF
    }

    #[inline]
    pub const fn summary(self) -> u32 {
        (self.0 >> 21) & 0x3F
    }

    #[inline]
    pub const fn level(self) -> u32 {
        (self.0 >> 27) & 0x1F
    }
}

impl fmt::Debug for ResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResultCode(0x{:08X} level={} summary={} module={} desc={})",
            self.0,
            self.level(),
            self.summary(),
            self.module(),
            self.description()
        )
    }
}

impl fmt::Display for ResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08X}", self.0)
    }
}

/// error codes the HLE kernel and services return often enough to be worth
/// naming.
pub mod errors {
    use super::*;

    // --- Kernel -----------------------------------------------------------
    /// 0xD8E007F7
    pub const INVALID_HANDLE: ResultCode =
        ResultCode::new(Level::Permanent, Summary::InvalidArgument, Module::Kernel, 1015);
    /// 0xD86007F3
    pub const OUT_OF_MEMORY: ResultCode =
        ResultCode::new(Level::Permanent, Summary::OutOfResource, Module::Kernel, 1011);
    /// 0xD88007FA
    pub const NOT_FOUND: ResultCode =
        ResultCode::new(Level::Permanent, Summary::NotFound, Module::Kernel, 1018);
    /// 0xD8E00405
    pub const INVALID_ENUM_VALUE: ResultCode =
        ResultCode::new(Level::Permanent, Summary::InvalidArgument, Module::Kernel, 5);

    // --- OS ---------------------------------------------------------------
    /// 0xE0E01BF5
    pub const INVALID_ADDRESS: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 1013);
    /// 0xE0E01BF1
    pub const MISALIGNED_ADDRESS: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 1009);
    /// 0xE0E01BF2
    pub const MISALIGNED_SIZE: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 1010);
    /// 0xE0E01BEE
    pub const INVALID_COMBINATION: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 1006);
    /// 0xE0E01BFD
    pub const OUT_OF_RANGE: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 1021);
    /// 0x09401BFE, svcWaitSynchronization timing out is *not* a failure.
    pub const TIMEOUT: ResultCode =
        ResultCode::new(Level::Info, Summary::StatusChanged, Module::Os, 1022);
    /// 0xC920181A, the other end of a session went away.
    pub const SESSION_CLOSED: ResultCode =
        ResultCode::new(Level::Status, Summary::Canceled, Module::Os, 26);
    /// 0xE0E0181E
    pub const PORT_NAME_TOO_LONG: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Os, 30);

    // --- FS ---------------------------------------------------------------
    /// 0xC8804478
    pub const FS_NOT_FOUND: ResultCode =
        ResultCode::new(Level::Status, Summary::NotFound, Module::Fs, 120);
    /// 0xC8804470
    pub const FS_FILE_NOT_FOUND: ResultCode =
        ResultCode::new(Level::Status, Summary::NotFound, Module::Fs, 112);
    /// 0xC8804471, a directory on the way to the file is missing.
    pub const FS_PATH_NOT_FOUND: ResultCode =
        ResultCode::new(Level::Status, Summary::NotFound, Module::Fs, 113);
    /// 0xC8A04478, what opening extra data that was never created gives.
    pub const FS_NOT_FOUND_INVALID_STATE: ResultCode =
        ResultCode::new(Level::Status, Summary::InvalidState, Module::Fs, 120);
    /// 0xC8A04554, a save archive that was never formatted.
    pub const FS_NOT_FORMATTED: ResultCode =
        ResultCode::new(Level::Status, Summary::InvalidState, Module::Fs, 340);
    /// 0xC82044B4
    pub const FS_FILE_ALREADY_EXISTS: ResultCode =
        ResultCode::new(Level::Status, Summary::NothingHappened, Module::Fs, 180);
    /// 0xC82044B9
    pub const FS_DIRECTORY_ALREADY_EXISTS: ResultCode =
        ResultCode::new(Level::Status, Summary::NothingHappened, Module::Fs, 185);
    /// 0xC92044F0
    pub const FS_DIRECTORY_NOT_EMPTY: ResultCode =
        ResultCode::new(Level::Status, Summary::Canceled, Module::Fs, 240);
    /// 0xE0C04702, a file where a directory was expected, or the reverse.
    pub const FS_UNEXPECTED_FILE_OR_DIRECTORY: ResultCode =
        ResultCode::new(Level::Usage, Summary::NotSupported, Module::Fs, 770);
    /// 0xE0E046BE
    pub const FS_INVALID_PATH: ResultCode =
        ResultCode::new(Level::Usage, Summary::InvalidArgument, Module::Fs, 702);
    /// 0xC8A0445A
    pub const FS_ARCHIVE_NOT_MOUNTED: ResultCode =
        ResultCode::new(Level::Status, Summary::InvalidState, Module::Fs, 90);
    /// 0xC82044BE
    pub const FS_ALREADY_EXISTS: ResultCode =
        ResultCode::new(Level::Status, Summary::NothingHappened, Module::Fs, 190);
    /// 0xD960442C
    pub const FS_NOT_IMPLEMENTED: ResultCode =
        ResultCode::new(Level::Permanent, Summary::Internal, Module::Fs, 44);

    // --- UDS --------------------------------------------------------------
    /// 0xC9411002, what nwm::UDS answers on a console whose wireless is
    /// switched off, and to anything sent to it before it was initialized.
    pub const UDS_WIRELESS_OFF: ResultCode =
        ResultCode::new(Level::Status, Summary::StatusChanged, Module::Uds, 2);

    /// generic "this HLE stub does not exist yet".
    pub const UNIMPLEMENTED: ResultCode =
        ResultCode::new(Level::Permanent, Summary::NotSupported, Module::Common, 1023);

    /// not a real hardware code either, but a deliberately chosen one, "this
    /// online feature is unavailable" for the network-facing services we have
    /// not implemented (frd:u, boss:U, http:C, ...). local wireless answers
    /// with [UDS_WIRELESS_OFF] instead.
    pub const NOT_CONNECTED: ResultCode =
        ResultCode::new(Level::Status, Summary::NotFound, Module::Ac, 12);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_fields_in_the_right_places() {
        let r = ResultCode::new(Level::Permanent, Summary::InvalidArgument, Module::Kernel, 1015);
        assert_eq!(r.0, 0xD8E0_07F7);
        assert_eq!(r.level(), 27);
        assert_eq!(r.summary(), 7);
        assert_eq!(r.module(), 1);
        assert_eq!(r.description(), 1015);
    }

    #[test]
    fn success_is_zero() {
        assert!(RESULT_SUCCESS.is_success());
        assert!(!RESULT_SUCCESS.is_error());
    }

    /// games compare against these literals, so a typo in the enum combination
    /// would be a silent behavior change. Pin every one.
    #[test]
    fn well_known_codes_match_hardware() {
        use errors::*;
        for (name, got, want) in [
            ("INVALID_HANDLE", INVALID_HANDLE.0, 0xD8E0_07F7u32),
            ("OUT_OF_MEMORY", OUT_OF_MEMORY.0, 0xD860_07F3),
            ("NOT_FOUND", NOT_FOUND.0, 0xD880_07FA),
            ("INVALID_ENUM_VALUE", INVALID_ENUM_VALUE.0, 0xD8E0_0405),
            ("INVALID_ADDRESS", INVALID_ADDRESS.0, 0xE0E0_1BF5),
            ("MISALIGNED_ADDRESS", MISALIGNED_ADDRESS.0, 0xE0E0_1BF1),
            ("MISALIGNED_SIZE", MISALIGNED_SIZE.0, 0xE0E0_1BF2),
            ("INVALID_COMBINATION", INVALID_COMBINATION.0, 0xE0E0_1BEE),
            ("OUT_OF_RANGE", OUT_OF_RANGE.0, 0xE0E0_1BFD),
            ("TIMEOUT", TIMEOUT.0, 0x0940_1BFE),
            ("SESSION_CLOSED", SESSION_CLOSED.0, 0xC920_181A),
            ("PORT_NAME_TOO_LONG", PORT_NAME_TOO_LONG.0, 0xE0E0_181E),
            ("FS_NOT_FOUND", FS_NOT_FOUND.0, 0xC880_4478),
            ("FS_ARCHIVE_NOT_MOUNTED", FS_ARCHIVE_NOT_MOUNTED.0, 0xC8A0_445A),
            ("FS_FILE_NOT_FOUND", FS_FILE_NOT_FOUND.0, 0xC880_4470),
            ("FS_PATH_NOT_FOUND", FS_PATH_NOT_FOUND.0, 0xC880_4471),
            ("FS_NOT_FOUND_INVALID_STATE", FS_NOT_FOUND_INVALID_STATE.0, 0xC8A0_4478),
            ("FS_NOT_FORMATTED", FS_NOT_FORMATTED.0, 0xC8A0_4554),
            ("FS_FILE_ALREADY_EXISTS", FS_FILE_ALREADY_EXISTS.0, 0xC820_44B4),
            ("FS_DIRECTORY_ALREADY_EXISTS", FS_DIRECTORY_ALREADY_EXISTS.0, 0xC820_44B9),
            ("FS_DIRECTORY_NOT_EMPTY", FS_DIRECTORY_NOT_EMPTY.0, 0xC920_44F0),
            ("FS_UNEXPECTED_FILE_OR_DIRECTORY", FS_UNEXPECTED_FILE_OR_DIRECTORY.0, 0xE0C0_4702),
            ("FS_INVALID_PATH", FS_INVALID_PATH.0, 0xE0E0_46BE),
            ("FS_ALREADY_EXISTS", FS_ALREADY_EXISTS.0, 0xC820_44BE),
            ("FS_NOT_IMPLEMENTED", FS_NOT_IMPLEMENTED.0, 0xD960_442C),
            ("UDS_WIRELESS_OFF", UDS_WIRELESS_OFF.0, 0xC941_1002),
        ] {
            assert_eq!(got, want, "{name}: got 0x{got:08X} want 0x{want:08X}");
        }
    }
}
