//! keyboard to 3DS button mapping.

use winit::keyboard::{KeyCode, PhysicalKey};
use zakuro_core::services::hid::{InputState, PadState};

/// a default layout that works on a keyboard without thinking about it.
pub fn button_for(key: KeyCode) -> Option<PadState> {
    Some(match key {
        KeyCode::KeyX => PadState::A,
        KeyCode::KeyZ => PadState::B,
        KeyCode::KeyS => PadState::X,
        KeyCode::KeyA => PadState::Y,
        KeyCode::KeyQ => PadState::L,
        KeyCode::KeyW => PadState::R,
        KeyCode::Enter => PadState::START,
        KeyCode::Backspace => PadState::SELECT,
        KeyCode::ArrowUp => PadState::UP,
        KeyCode::ArrowDown => PadState::DOWN,
        KeyCode::ArrowLeft => PadState::LEFT,
        KeyCode::ArrowRight => PadState::RIGHT,
        _ => return None,
    })
}

/// circle pad axes, as (x, y) contributions.
pub fn circle_for(key: KeyCode) -> Option<(f32, f32)> {
    Some(match key {
        KeyCode::KeyI => (0.0, 1.0),
        KeyCode::KeyK => (0.0, -1.0),
        KeyCode::KeyJ => (-1.0, 0.0),
        KeyCode::KeyL => (1.0, 0.0),
        _ => return None,
    })
}

/// accumulates key state between frames.
#[derive(Default)]
pub struct Keyboard {
    buttons: PadState,
    circle: (f32, f32),
    touch: Option<(u16, u16)>,
}

impl Keyboard {
    pub fn key(&mut self, key: PhysicalKey, pressed: bool) {
        let PhysicalKey::Code(code) = key else {
            return;
        };
        if let Some(button) = button_for(code) {
            self.buttons.set(button, pressed);
        }
        if let Some((x, y)) = circle_for(code) {
            // holding two opposite keys cancels out, which is what a real
            // stick would do.
            if pressed {
                self.circle.0 += x;
                self.circle.1 += y;
            } else {
                self.circle.0 -= x;
                self.circle.1 -= y;
            }
            self.circle.0 = self.circle.0.clamp(-1.0, 1.0);
            self.circle.1 = self.circle.1.clamp(-1.0, 1.0);
        }
    }

    /// records a click on the bottom screen, in screen pixels, or its release.
    pub fn touch(&mut self, position: Option<(u16, u16)>) {
        self.touch = position;
    }

    pub fn state(&self) -> InputState {
        InputState {
            buttons: self.buttons,
            circle_x: self.circle.0,
            circle_y: self.circle.1,
            touch: self.touch,
        }
    }
}
