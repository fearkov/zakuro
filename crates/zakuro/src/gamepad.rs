//! game controllers, any the system knows, read through gilrs, and the
//! Switch's Pro Controller on macOS read from its reports, see
//! pro_controller.

use gilrs::{Axis, Button, EventType, Gamepad, Gilrs};
use zakuro_core::services::hid::{InputState, PadState};

use crate::pro_controller::{is_pro_controller, Controllers, Report};
use crate::settings::PadButtons;

/// how far a stick has to lean before it counts, some rest a little off
/// center.
const DEADZONE: f32 = 0.15;

pub struct Gamepads {
    /// none when the system could not be asked for controllers.
    gilrs: Option<Gilrs>,
    buttons: PadButtons,
    /// Pro Controllers read from their own reports, on macOS.
    pro: Controllers,
    /// the buttons they held at the last poll, to tell the ones pressed
    /// since.
    pro_held: Vec<Button>,
}

/// whether gilrs's reading of a controller is left out, a Pro Controller's
/// while one is read from its reports.
fn left_out(pad: &Gamepad, reports: &[Report]) -> bool {
    !reports.is_empty() && is_pro_controller(pad.vendor_id(), pad.product_id(), pad.name())
}

impl Gamepads {
    pub fn new(buttons: PadButtons) -> Gamepads {
        let gilrs = Gilrs::new()
            .inspect_err(|error| log::warn!("controllers are not available, {error}"))
            .ok();
        Gamepads { gilrs, buttons, pro: Controllers::new(), pro_held: Vec::new() }
    }

    pub fn set_buttons(&mut self, buttons: PadButtons) {
        self.buttons = buttons;
    }

    /// takes in what the controllers did since the last call, and returns the
    /// buttons pressed meanwhile.
    pub fn poll(&mut self) -> Vec<Button> {
        let mut pressed = Vec::new();
        let reports = self.pro.reports();
        if let Some(gilrs) = &mut self.gilrs {
            while let Some(event) = gilrs.next_event() {
                match event.event {
                    EventType::ButtonPressed(button, _) if !left_out(&gilrs.gamepad(event.id), &reports) => pressed.push(button),
                    EventType::Connected => {
                        log::info!("controller connected, {}", gilrs.gamepad(event.id).name());
                    }
                    EventType::Disconnected => log::info!("controller disconnected"),
                    _ => {}
                }
            }
        }
        let held: Vec<Button> = reports.iter().flat_map(|report| report.buttons.iter().copied()).collect();
        pressed.extend(held.iter().filter(|button| !self.pro_held.contains(button)));
        self.pro_held = held;
        pressed
    }

    /// whether a controller holds button down.
    pub fn holding(&self, button: Button) -> bool {
        let reports = self.pro.reports();
        reports.iter().any(|report| report.buttons.contains(&button))
            || self.gilrs.as_ref().is_some_and(|gilrs| {
                gilrs.gamepads().any(|(_, pad)| !left_out(&pad, &reports) && pad.is_pressed(button))
            })
    }

    /// what the controllers hold down, over what the keyboard does.
    pub fn apply(&self, mut state: InputState) -> InputState {
        let reports = self.pro.reports();
        for report in &reports {
            for (button, bound) in self.buttons.map() {
                if report.buttons.contains(&bound) {
                    state.buttons.set(button, true);
                }
            }
            sticks(&mut state, report.left, report.right);
        }
        let Some(gilrs) = &self.gilrs else { return state };
        for (_, pad) in gilrs.gamepads() {
            if left_out(&pad, &reports) {
                continue;
            }
            for (button, bound) in self.buttons.map() {
                if pad.is_pressed(bound) {
                    state.buttons.set(button, true);
                }
            }
            // the d-pad of some controllers is a pair of axes
            let dpad = [
                (PadState::RIGHT, pad.value(Axis::DPadX) > 0.5),
                (PadState::LEFT, pad.value(Axis::DPadX) < -0.5),
                (PadState::UP, pad.value(Axis::DPadY) > 0.5),
                (PadState::DOWN, pad.value(Axis::DPadY) < -0.5),
            ];
            for (button, held) in dpad {
                if held {
                    state.buttons.set(button, true);
                }
            }
            let left = [pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY)];
            let right = [pad.value(Axis::RightStickX), pad.value(Axis::RightStickY)];
            sticks(&mut state, left, right);
        }
        state
    }
}

/// the left stick is the circle pad and the right one the C-stick, up
/// positive as the console has it, each once it leans past the deadzone.
fn sticks(state: &mut InputState, [x, y]: [f32; 2], [cx, cy]: [f32; 2]) {
    if x.hypot(y) > DEADZONE {
        state.circle_x = x.clamp(-1.0, 1.0);
        state.circle_y = y.clamp(-1.0, 1.0);
    }
    if cx.hypot(cy) > DEADZONE {
        state.c_stick_x = cx.clamp(-1.0, 1.0);
        state.c_stick_y = cy.clamp(-1.0, 1.0);
    }
}

/// a controller button as a person would call it.
pub fn button_name(button: Button) -> &'static str {
    match button {
        Button::South => "Bottom face",
        Button::East => "Right face",
        Button::North => "Top face",
        Button::West => "Left face",
        Button::LeftTrigger => "LB / L1",
        Button::RightTrigger => "RB / R1",
        Button::LeftTrigger2 => "LT / L2",
        Button::RightTrigger2 => "RT / R2",
        Button::Select => "Back / Select",
        Button::Start => "Start",
        Button::Mode => "Home",
        Button::LeftThumb => "Left stick",
        Button::RightThumb => "Right stick",
        Button::DPadUp => "D-pad up",
        Button::DPadDown => "D-pad down",
        Button::DPadLeft => "D-pad left",
        Button::DPadRight => "D-pad right",
        Button::C => "C",
        Button::Z => "Z",
        Button::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// with no controller, or no way to ask for one, the keyboard's input
    /// goes through as it was.
    #[test]
    fn no_controller_leaves_the_keyboard_alone() {
        let gamepads = Gamepads::new(PadButtons::default());
        if gamepads.gilrs.as_ref().is_some_and(|gilrs| gilrs.gamepads().next().is_some()) {
            return;
        }
        let state = InputState { buttons: PadState::A, circle_x: 1.0, circle_y: 0.0, touch: Some((1, 2)), ..InputState::default() };
        let applied = gamepads.apply(state);
        assert_eq!(applied.buttons, PadState::A);
        assert_eq!((applied.circle_x, applied.circle_y, applied.touch), (1.0, 0.0, Some((1, 2))));
    }
}
