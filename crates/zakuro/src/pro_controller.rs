//! the Switch's Pro Controller, and the ones made like it, in the mode
//! macOS puts it in. its full reports, 0x30 and the like, are not what its
//! HID descriptor says, and reading them by it takes the report's timer for
//! buttons, which then count up on their own, and leans the left stick up.
//! on macOS such a controller is read here, straight from its reports, and
//! gilrs's reading of it is left out while they come. nothing is sent to
//! it, the system goes on driving it.

// the reports are read on macOS alone, elsewhere only the tests parse them
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use gilrs::Button;

pub const NINTENDO: u16 = 0x057E;
pub const PRO_CONTROLLER: u16 = 0x2009;
/// what the controllers made like it go by.
const NAME: &str = "Pro Controller";

/// whether a controller is a Pro Controller, or made like one, by its ids
/// or its name.
pub fn is_pro_controller(vendor: Option<u16>, product: Option<u16>, name: &str) -> bool {
    (vendor == Some(NINTENDO) && product == Some(PRO_CONTROLLER)) || name == NAME
}

/// the buttons in a full report, the byte after the report id each is in,
/// its bit and the button it is, by where it sits as gilrs names them. the
/// Switch's A, B, X and Y are where the 3DS has them.
const BUTTONS: [(usize, u8, Button); 17] = [
    (3, 0x01, Button::West),
    (3, 0x02, Button::North),
    (3, 0x04, Button::South),
    (3, 0x08, Button::East),
    (3, 0x40, Button::RightTrigger),
    (3, 0x80, Button::RightTrigger2),
    (4, 0x01, Button::Select),
    (4, 0x02, Button::Start),
    (4, 0x04, Button::RightThumb),
    (4, 0x08, Button::LeftThumb),
    (4, 0x10, Button::Mode),
    (5, 0x01, Button::DPadDown),
    (5, 0x02, Button::DPadUp),
    (5, 0x04, Button::DPadRight),
    (5, 0x08, Button::DPadLeft),
    (5, 0x40, Button::LeftTrigger),
    (5, 0x80, Button::LeftTrigger2),
];

/// a stick at rest, in the 12 bits each of its axes takes.
const CENTER: f32 = 2048.0;
/// how far a stick is taken to go until it is seen to go further, short of
/// any controller's, so that a full push reaches the edge.
const FIRST_REACH: f32 = 1000.0;

/// what a full report says: the buttons held, and the left and right
/// sticks, x and y from -1 to 1, up positive.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub buttons: Vec<Button>,
    pub left: [f32; 2],
    pub right: [f32; 2],
}

/// how far each stick axis has been seen to go from the center, which a
/// full push of it is.
#[derive(Debug, Clone)]
pub struct Calibration {
    reach: [f32; 4],
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration { reach: [FIRST_REACH; 4] }
    }
}

impl Calibration {
    fn axis(&mut self, index: usize, raw: u16) -> f32 {
        let offset = raw as f32 - CENTER;
        let reach = &mut self.reach[index];
        *reach = reach.max(offset.abs());
        (offset / *reach).clamp(-1.0, 1.0)
    }
}

/// what a report says, none when it is not a full one.
pub fn parse(report: &[u8], calibration: &mut Calibration) -> Option<Report> {
    // the full reports, the standard one, the ones with more after it and a
    // subcommand's reply, have the buttons and sticks in the same place
    if !matches!(report.first(), Some(0x21 | 0x30 | 0x31 | 0x32 | 0x33)) || report.len() < 12 {
        return None;
    }
    let buttons = BUTTONS.iter().filter(|&&(at, bit, _)| report[at] & bit != 0).map(|&(.., button)| button).collect();
    // two axes of 12 bits in three bytes, the second's low four bits after
    // the first's eight
    let stick = |at: usize| {
        let (a, b, c) = (report[at] as u16, report[at + 1] as u16, report[at + 2] as u16);
        (a | (b & 0x0F) << 8, b >> 4 | c << 4)
    };
    let ((lx, ly), (rx, ry)) = (stick(6), stick(9));
    Some(Report {
        buttons,
        left: [calibration.axis(0, lx), calibration.axis(1, ly)],
        right: [calibration.axis(2, rx), calibration.axis(3, ry)],
    })
}

/// the Pro Controllers read from their reports, the latest of each.
#[derive(Default)]
pub struct Controllers {
    #[cfg(target_os = "macos")]
    latest: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, (Report, std::time::Instant)>>>,
}

impl Controllers {
    pub fn new() -> Controllers {
        let controllers = Controllers::default();
        controllers.watch();
        controllers
    }

    /// starts looking for them, on macOS only, where the system puts them
    /// in the mode gilrs cannot read.
    fn watch(&self) {
        #[cfg(target_os = "macos")]
        {
            let latest = self.latest.clone();
            if let Err(error) = std::thread::Builder::new().name("pro controllers".into()).spawn(move || reader::watch(&latest)) {
                log::warn!("could not start looking for Pro Controllers, {error}");
            }
        }
    }

    /// the latest report of each controller sending full reports now.
    pub fn reports(&self) -> Vec<Report> {
        #[cfg(target_os = "macos")]
        {
            // a controller whose reports stopped, gone or back to the
            // simple mode gilrs reads, is gilrs's again
            let fresh = std::time::Duration::from_millis(500);
            if let Ok(latest) = self.latest.lock() {
                return latest.values().filter(|(_, at)| at.elapsed() < fresh).map(|(report, _)| report.clone()).collect();
            }
        }
        Vec::new()
    }
}

#[cfg(target_os = "macos")]
mod reader {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use super::{is_pro_controller, parse, Calibration, Report};

    type Latest = Arc<Mutex<BTreeMap<String, (Report, Instant)>>>;

    /// opens each Pro Controller as it shows up, shared with the system and
    /// gilrs, and reads it on a thread of its own.
    pub fn watch(latest: &Latest) {
        let mut api = match hidapi::HidApi::new() {
            Ok(api) => api,
            Err(error) => {
                log::warn!("controllers can't be read directly, {error}");
                return;
            }
        };
        let mut reading: BTreeMap<String, Arc<AtomicBool>> = BTreeMap::new();
        loop {
            reading.retain(|_, alive| alive.load(Ordering::Relaxed));
            for info in api.device_list() {
                if !is_pro_controller(Some(info.vendor_id()), Some(info.product_id()), info.product_string().unwrap_or("")) {
                    continue;
                }
                let path = info.path().to_string_lossy().into_owned();
                if reading.contains_key(&path) {
                    continue;
                }
                match api.open_path(info.path()) {
                    Ok(device) => {
                        let alive = Arc::new(AtomicBool::new(true));
                        let (latest, keep, name) = (latest.clone(), alive.clone(), path.clone());
                        let started = std::thread::Builder::new().name("pro controller".into()).spawn(move || read(device, &name, &latest, &keep));
                        match started {
                            Ok(_) => {
                                log::info!("reading a Pro Controller from its reports");
                                reading.insert(path, alive);
                            }
                            Err(error) => log::warn!("could not start reading a Pro Controller, {error}"),
                        }
                    }
                    Err(error) => log::debug!("a Pro Controller could not be opened, {error}"),
                }
            }
            std::thread::sleep(Duration::from_secs(2));
            if let Err(error) = api.refresh_devices() {
                log::debug!("could not look for controllers again, {error}");
            }
        }
    }

    /// reads a controller until it goes away.
    fn read(device: hidapi::HidDevice, path: &str, latest: &Latest, alive: &AtomicBool) {
        let mut calibration = Calibration::default();
        let mut buffer = [0u8; 64];
        loop {
            match device.read_timeout(&mut buffer, 500) {
                Ok(0) => {}
                Ok(length) => {
                    if let Some(report) = parse(&buffer[..length], &mut calibration) {
                        if let Ok(mut latest) = latest.lock() {
                            latest.insert(path.to_owned(), (report, Instant::now()));
                        }
                    }
                }
                Err(error) => {
                    log::info!("a Pro Controller went away, {error}");
                    break;
                }
            }
        }
        if let Ok(mut latest) = latest.lock() {
            latest.remove(path);
        }
        alive.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a full report with the buttons and the two sticks given.
    fn report(timer: u8, buttons: [u8; 3], left: (u16, u16), right: (u16, u16)) -> Vec<u8> {
        let stick = |(x, y): (u16, u16)| [x as u8, ((x >> 8) & 0x0F) as u8 | ((y & 0x0F) << 4) as u8, (y >> 4) as u8];
        let mut bytes = vec![0x30, timer, 0x8E];
        bytes.extend_from_slice(&buttons);
        bytes.extend_from_slice(&stick(left));
        bytes.extend_from_slice(&stick(right));
        bytes.resize(49, 0);
        bytes
    }

    /// the report's timer counting up is no button, which is what read by
    /// the descriptor counts up as buttons 1 to 8, and the sticks at rest
    /// are at rest.
    #[test]
    fn the_timer_is_no_button() {
        let mut calibration = Calibration::default();
        for timer in 0..=255 {
            let parsed = parse(&report(timer, [0; 3], (2048, 2048), (2048, 2048)), &mut calibration).unwrap();
            assert_eq!(parsed, Report::default(), "timer {timer}");
        }
    }

    /// A on the right byte, minus and home on the shared one, up and ZL on
    /// the left one.
    #[test]
    fn buttons_come_out_where_they_sit() {
        let mut calibration = Calibration::default();
        let parsed = parse(&report(7, [0x08, 0x11, 0x82], (2048, 2048), (2048, 2048)), &mut calibration).unwrap();
        assert_eq!(parsed.buttons, [Button::East, Button::Select, Button::Mode, Button::DPadUp, Button::LeftTrigger2]);
    }

    /// a stick pushed up and right reads up and right, a full push reaches
    /// the edge, and one pushed further later is the new edge.
    #[test]
    fn sticks_lean_the_way_they_are_pushed() {
        let mut calibration = Calibration::default();
        let parsed = parse(&report(0, [0; 3], (2048 + 500, 2048 + 1000), (2048 - 1000, 2048)), &mut calibration).unwrap();
        assert_eq!(parsed.left, [0.5, 1.0]);
        assert_eq!(parsed.right, [-1.0, 0.0]);
        let parsed = parse(&report(1, [0; 3], (2048 + 1500, 2048 + 750), (2048, 2048)), &mut calibration).unwrap();
        assert_eq!(parsed.left, [1.0, 0.75]);
    }

    /// the simple mode's reports are gilrs's to read.
    #[test]
    fn simple_reports_are_left_alone() {
        let mut calibration = Calibration::default();
        assert_eq!(parse(&[0x3F, 0x08, 0, 8, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80], &mut calibration), None);
        assert_eq!(parse(&[0x30, 0], &mut calibration), None);
    }

    #[test]
    fn pro_controllers_are_told_by_ids_or_name() {
        assert!(is_pro_controller(Some(NINTENDO), Some(PRO_CONTROLLER), "Wireless Gamepad"));
        assert!(is_pro_controller(Some(0x1234), Some(0x5678), "Pro Controller"));
        assert!(!is_pro_controller(Some(0x045E), Some(0x02EA), "Xbox Wireless Controller"));
    }
}
