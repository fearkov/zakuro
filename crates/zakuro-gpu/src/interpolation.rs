//! frame interpolation, a picture in between each two of a title's own, for
//! titles that draw one every other refresh. the draws of a frame are
//! matched with the frame before's, by what they draw and how near their
//! places are, and each one that moved is drawn a second time with the float
//! uniforms of its shaders halfway between its own and its match's. that
//! draw goes into twins of the buffers it draws into, see the hardware
//! renderer, and the twin of a screen's buffer is the picture in between.
//! the matrices of a draw are mostly what changes, and halfway between two
//! of them is close to the transform halfway between, for the little a
//! thirtieth of a second moves things.

use std::collections::HashMap;

use crate::registers::*;
use crate::shader::{ShaderUnit, Vec4, FLOAT_UNIFORMS};

/// the vertex shader's float uniforms at a draw.
pub(crate) type Uniforms = [Vec4; FLOAT_UNIFORMS];

/// how much a component's move from one frame to the next may differ from
/// its move the frame before, before the draw counts as a different one, a
/// jump rather than a move, which is drawn as it is. a model in the distance
/// or the camera going on as it went moves as it did however fast, a camera
/// cutting elsewhere, another model taken for this one or a flag kept in a
/// float do not. a component of a rotation turning faster by less than
/// about 30 degrees moves less than half again, a translation less than a
/// quarter of its distance again.
const JUMP: f32 = 0.5;
const JUMP_PART: f32 = 0.25;

/// whether a component a texture's coordinates may read steps from one
/// round value to another, sixty-fourths, as a sprite's offset into its
/// sheet of them does, rather than moves, as a matrix's does.
fn steps(before: f32, now: f32) -> bool {
    let round = |value: f32| (value * 64.0).fract() == 0.0;
    round(before) && round(now)
}

/// how many draws of the frame before drawn the same way a draw's match is
/// looked for among, around its own place among them.
const NEAREST: usize = 32;

/// for how many frames after jumping a draw that jumps again counts as one
/// that keeps jumping rather than as a camera cutting elsewhere. a sprite's
/// animation that steps every other frame jumps every other frame.
const RESTLESS: u8 = 4;

/// the draws of a frame drawn the same way, see key, in the order they
/// came, how far each one's uniforms moved since the frame before, zero
/// for one it had not, how many frames on a jump still counts as one more
/// of its jumps, see RESTLESS, and which of them a draw of the next frame
/// took for its own.
#[derive(Default)]
struct Drawn {
    uniforms: Vec<Uniforms>,
    moves: Vec<Uniforms>,
    restless: Vec<u8>,
    taken: Vec<bool>,
}

/// the draw of the frame before, among those drawn the same way and not
/// taken yet, whose uniforms placing the vertices are nearest a draw's,
/// around its place among them, the one at its place when that is as near.
/// a title can draw a model twice in an order of its own each frame,
/// Majora's Mask 3D draws the two eyes of the 3D the other way round every
/// other frame, and in the order they came they would swap places.
fn nearest(drawn: &Drawn, place: usize, now: &Uniforms, placing: u128) -> Option<usize> {
    let distance = |before: &Uniforms| {
        let mut total = 0.0f32;
        let mut uniforms = placing;
        while uniforms != 0 {
            let uniform = uniforms.trailing_zeros() as usize;
            uniforms &= uniforms - 1;
            total += before[uniform].iter().zip(&now[uniform]).map(|(before, now)| (before - now).abs()).sum::<f32>();
        }
        // NaN as far as anything gets
        if total.is_nan() { f32::INFINITY } else { total }
    };
    let count = drawn.uniforms.len();
    let around = match count <= NEAREST {
        true => 0..count,
        false => place.saturating_sub(NEAREST / 2)..(place + NEAREST / 2).min(count),
    };
    let own = (place < count && !drawn.taken[place]).then(|| (place, distance(&drawn.uniforms[place])));
    if own.is_some_and(|(_, distance)| distance == 0.0) {
        return own.map(|(place, _)| place);
    }
    let others = around.filter(|&i| i != place && !drawn.taken[i]).map(|i| (i, distance(&drawn.uniforms[i])));
    own.into_iter().chain(others).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i)
}

/// what frame interpolation keeps of the draws of a frame and the one before.
#[derive(Default)]
pub(crate) struct Interpolation {
    /// the uniforms of the frame before's draws, by what they drew.
    before: HashMap<u64, Drawn>,
    /// this frame's so far.
    now: HashMap<u64, Drawn>,
    /// uniforms in between given back, for the next ones rather than
    /// allocating.
    spare: Option<Box<Uniforms>>,
    /// the uniforms each program places its vertices with, by its
    /// fingerprint, its entry point, the outputs that place them and those
    /// of the textures' coordinates, see ShaderUnit::uniforms_feeding, those
    /// it may read for the coordinates too second. only those move in
    /// between, a texture's offset into a sheet of sprites or a flag are as
    /// the frame's.
    placing: HashMap<(u64, u32, u16, u16), (u128, u128)>,
    /// this frame's counts, see Matches.
    draws: u32,
    matched: u32,
    moved: u32,
    jumped: u32,
    jumping: u32,
}

/// how the draws of a frame went, see Interpolation::frame_done. of those
/// the frame before had too, those that moved, those that jumped, and those
/// that jumped a few frames before too, see RESTLESS, as a batch of sprites
/// sorted anew every frame or a flag that flips does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Matches {
    pub draws: u32,
    pub matched: u32,
    pub moved: u32,
    pub jumped: u32,
    pub jumping: u32,
}

impl Matches {
    /// whether the picture in between looks right. most of the draws were
    /// there the frame before, and few of those that had not jumped lately
    /// jumped now, as everything does when the camera cuts to somewhere else.
    pub fn smooth(&self) -> bool {
        self.moved > 0 && self.matched * 2 >= self.draws && self.jumped * 4 <= self.matched
    }
}

impl Interpolation {
    /// the uniforms halfway between the draw's and those of the frame
    /// before's draw that matches it, for drawing it again in between, none
    /// when they are the same or it has no match or it jumped, or nothing
    /// is drawn in between. the draw's own are kept for the next frame's to
    /// match. a draw through a geometry shader is left as it is. give the
    /// uniforms back once drawn with.
    pub(crate) fn draw(&mut self, registers: &[u32], vertex: &ShaderUnit, geometry: Option<&ShaderUnit>, between: bool) -> Option<Box<Uniforms>> {
        self.draws += 1;
        let key = key(registers, vertex, geometry);
        let drawn = self.now.entry(key).or_default();
        drawn.uniforms.push(*vertex.float_uniforms);
        drawn.moves.push([[0.0; 4]; FLOAT_UNIFORMS]);
        drawn.restless.push(0);
        let place = drawn.uniforms.len() - 1;
        if !between {
            return None;
        }
        if geometry.is_some() {
            self.matched += self.before.get(&key).is_some_and(|before| place < before.uniforms.len()) as u32;
            return None;
        }
        let (outputs, coordinates) = crate::raster::placing_outputs(registers);
        let (placing, shared) = *self
            .placing
            .entry((vertex.fingerprint(), vertex.entry_point, outputs, coordinates))
            .or_insert_with(|| vertex.uniforms_feeding(outputs, coordinates));
        let drawn = self.before.get_mut(&key)?;
        let now = self.now.get_mut(&key)?;
        let found = nearest(drawn, place, &now.uniforms[place], placing | shared)?;
        drawn.taken[found] = true;
        let (before, moved_before, restless) = (&drawn.uniforms[found], &drawn.moves[found], drawn.restless[found]);
        let (now, moved_now, still_restless) = (&now.uniforms[place], &mut now.moves[place], &mut now.restless[place]);
        *still_restless = restless.saturating_sub(1);
        for ((moved, now), before) in moved_now.iter_mut().zip(now).zip(before) {
            *moved = std::array::from_fn(|i| now[i] - before[i]);
        }
        self.matched += 1;
        let placed = |uniform: usize| (placing | shared) & (1 << uniform) != 0;
        let shared = |uniform: usize| shared & (1 << uniform) != 0;
        // what moves, the components that changed of the uniforms placing
        // the vertices, but for steps of those the coordinates may read
        let moves = |uniform: usize, before: f32, now: f32| placed(uniform) && before != now && !(shared(uniform) && steps(before, now));
        let changes = |uniform: usize, before: &Vec4, now: &Vec4| (0..4).any(|i| moves(uniform, before[i], now[i]));
        if !before.iter().zip(now.iter()).enumerate().any(|(uniform, (before, now))| changes(uniform, before, now)) {
            return None;
        }
        let mut between = self.spare.take().unwrap_or_else(|| Box::new([[0.0; 4]; FLOAT_UNIFORMS]));
        let mut jumped = false;
        // a component jumps when its move differs from its move the frame
        // before by more than JUMP says. NaN and infinity compare false,
        // and jump
        let jumps = |before: f32, now: f32, moved_before: f32| {
            let change = (now - before - moved_before).abs();
            !(change <= JUMP || change <= JUMP_PART * before.abs().max(now.abs()))
        };
        let mut lerped = false;
        let uniforms = between.iter_mut().zip(before.iter()).zip(now.iter()).zip(moved_before.iter());
        for (uniform, (((out, before), now), moved_before)) in uniforms.enumerate() {
            let moving: [bool; 4] = std::array::from_fn(|i| moves(uniform, before[i], now[i]));
            let jumping: [bool; 4] = std::array::from_fn(|i| moving[i] && jumps(before[i], now[i], moved_before[i]));
            if jumping.contains(&true) {
                // one the coordinates may read too is as the frame has it,
                // whatever it is, one that only places the vertices says
                // the draw is another than the one before
                if !shared(uniform) {
                    jumped = true;
                    break;
                }
                *out = *now;
                continue;
            }
            *out = std::array::from_fn(|i| if moving[i] { before[i] + (now[i] - before[i]) * 0.5 } else { now[i] });
            lerped |= moving.contains(&true);
        }
        if !jumped && !lerped {
            // what moved jumped, and is as the frame has it
            self.spare = Some(between);
            return None;
        }
        if jumped {
            match restless {
                0 => self.jumped += 1,
                _ => self.jumping += 1,
            }
            *still_restless = RESTLESS;
            self.spare = Some(between);
            return None;
        }
        self.moved += 1;
        Some(between)
    }

    /// takes back the uniforms draw gave.
    pub(crate) fn give_back(&mut self, uniforms: Box<Uniforms>) {
        self.spare = Some(uniforms);
    }

    /// the frame is done, its draws are the ones the next frame's match,
    /// and how they matched the frame before's.
    pub(crate) fn frame_done(&mut self) -> Matches {
        let matches = Matches { draws: self.draws, matched: self.matched, moved: self.moved, jumped: self.jumped, jumping: self.jumping };
        let mut older = std::mem::replace(&mut self.before, std::mem::take(&mut self.now));
        for drawn in self.before.values_mut() {
            drawn.taken.clear();
            drawn.taken.resize(drawn.uniforms.len(), false);
        }
        // emptied, with the lists of the draws that frame had, which the
        // next one mostly draws too
        older.retain(|_, drawn| {
            let kept = !drawn.uniforms.is_empty();
            drawn.uniforms.clear();
            drawn.moves.clear();
            drawn.restless.clear();
            kept
        });
        self.now = older;
        (self.draws, self.matched, self.moved, self.jumped, self.jumping) = (0, 0, 0, 0, 0);
        matches
    }
}

/// what a draw draws, the same from frame to frame for a model drawn the
/// same way, its programs, its vertex arrays and the buffers it draws into
/// and its first texture.
fn key(registers: &[u32], vertex: &ShaderUnit, geometry: Option<&ShaderUnit>) -> u64 {
    let mix = |hash: u64, word: u64| (hash.rotate_left(5) ^ word).wrapping_mul(0x517C_C1B7_2722_0A95);
    // from an odd start, so that words of zero still count
    let mut hash = mix(mix(0x9E37_79B9_7F4A_7C15, vertex.fingerprint()), vertex.entry_point as u64);
    if let Some(geometry) = geometry {
        hash = mix(mix(mix(hash, 1), geometry.fingerprint()), geometry.entry_point as u64);
    }
    let arrays = REG_ATTRIBUTE_BASE..=REG_INDEX_ARRAY;
    let others = [
        REG_VERTEX_COUNT,
        REG_VERTEX_OFFSET,
        REG_PRIMITIVE_CONFIG,
        REG_GEOSTAGE_CONFIG,
        REG_COLOR_BUFFER_ADDRESS,
        REG_DEPTH_BUFFER_ADDRESS,
        REG_TEXTURE0_ADDRESS,
    ];
    for register in arrays.chain(others) {
        hash = mix(hash, registers[register] as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registers(vertices: u32) -> Vec<u32> {
        let mut registers = vec![0; 0x300];
        registers[REG_VERTEX_COUNT] = vertices;
        registers
    }

    fn unit(x: f32) -> ShaderUnit {
        let mut unit = ShaderUnit::new();
        unit.float_uniforms[0] = [1.0, 0.0, 0.0, x];
        unit.float_uniforms[1] = [0.0, 1.0, 0.0, 2.0];
        unit
    }

    /// a draw that moved between two frames is drawn in between halfway,
    /// one that stayed or is new is not drawn again, and draws of the same
    /// model match in the order they come.
    #[test]
    fn draws_match_the_frame_before_in_order() {
        let mut interpolation = Interpolation::default();
        assert!(interpolation.draw(&registers(3), &unit(0.0), None, true).is_none(), "nothing before");
        assert!(interpolation.draw(&registers(3), &unit(4.0), None, true).is_none());
        assert!(interpolation.draw(&registers(6), &unit(1.0), None, true).is_none());
        let first = interpolation.frame_done();
        assert_eq!(first, Matches { draws: 3, matched: 0, moved: 0, jumped: 0, jumping: 0 });
        assert!(!first.smooth());

        let between = interpolation.draw(&registers(3), &unit(0.5), None, true).expect("moved");
        assert_eq!(between[0], [1.0, 0.0, 0.0, 0.25]);
        assert_eq!(between[1], [0.0, 1.0, 0.0, 2.0]);
        interpolation.give_back(between);
        let between = interpolation.draw(&registers(3), &unit(4.2), None, true).expect("the second matches the second");
        assert!((between[0][3] - 4.1).abs() < 1e-6);
        interpolation.give_back(between);
        assert!(interpolation.draw(&registers(6), &unit(1.0), None, true).is_none(), "stayed");
        assert!(interpolation.draw(&registers(9), &unit(1.0), None, true).is_none(), "new");
        let second = interpolation.frame_done();
        assert_eq!(second, Matches { draws: 4, matched: 3, moved: 2, jumped: 0, jumping: 0 });
        assert!(second.smooth());
    }

    /// a uniform the coordinates may read too that jumps is as the frame has
    /// it, while what only places the vertices moves, and the draw does not
    /// count as a jump.
    #[test]
    fn a_shared_uniform_that_jumps_stays_as_the_frame_has_it() {
        // o0.x = c0 . v0, o0.y = c[a0.x + 20] . v0, o2 = c[a0.x + 20]
        let mut program = ShaderUnit::new();
        program.program[0] = (0x02 << 26) | (0x20 << 12);
        program.program[1] = (0x02 << 26) | (1 << 19) | ((0x20 + 20) << 12) | 1;
        program.program[2] = (0x13 << 26) | (0x02 << 21) | (1 << 19) | ((0x20 + 20) << 12);
        program.program[3] = 0x22 << 26;
        program.descriptors[0] = 0x8 | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        program.descriptors[1] = 0x4 | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        program.prepare();
        let mut registers = registers(3);
        registers[REG_SHADER_OUTPUT_TOTAL] = 3;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 2] = 0x1F1F_0D0C;
        let at = |x: f32, shared: f32| {
            let mut unit = program.clone();
            unit.float_uniforms[0] = [1.0, 0.0, 0.0, x];
            unit.float_uniforms[20] = [shared, 0.0, 0.0, 0.0];
            unit
        };
        let mut interpolation = Interpolation::default();
        for (x, shared) in [(0.0, 0.3), (0.1, 5.0)] {
            if let Some(between) = interpolation.draw(&registers, &at(x, shared), None, true) {
                interpolation.give_back(between);
            }
            interpolation.frame_done();
        }
        let between = interpolation.draw(&registers, &at(0.2, -5.0), None, true).expect("its place moved");
        assert!((between[0][3] - 0.15).abs() < 1e-6);
        assert_eq!(between[20][0], -5.0, "the shared one as the frame has it");
        interpolation.give_back(between);
        assert_eq!(interpolation.frame_done().jumped, 0);
        assert!(interpolation.draw(&registers, &at(0.2, 6.0), None, true).is_none(), "only the shared one changed, and jumped");
    }

    /// a model going on as it went moves in between however fast, through
    /// zero too, once it has moved a frame. its first move that fast jumps.
    #[test]
    fn going_on_as_before_moves_however_fast() {
        let mut interpolation = Interpolation::default();
        let mut moved = Vec::new();
        for x in [-16.3, -5.7, 5.1, 16.2, 27.5] {
            let between = interpolation.draw(&registers(3), &unit(x), None, true);
            moved.push(between.is_some());
            if let Some(between) = between {
                interpolation.give_back(between);
            }
            interpolation.frame_done();
        }
        assert_eq!(moved, [false, false, true, true, true], "nothing before, then the first move jumps");
    }

    /// a draw that jumps frame after frame, a batch of sprites sorted anew
    /// each frame, is drawn as it is and counts apart, the frame is still
    /// smooth for the rest of its draws. one that jumps after moving counts
    /// as a jump.
    #[test]
    fn draws_that_keep_jumping_count_apart() {
        let mut interpolation = Interpolation::default();
        let mut frame = |batch: f32, model: f32| {
            let batch = interpolation.draw(&registers(3), &unit(batch), None, true);
            let model = interpolation.draw(&registers(4), &unit(model), None, true);
            for between in [batch, model].into_iter().flatten() {
                interpolation.give_back(between);
            }
            interpolation.frame_done()
        };
        frame(0.0, 0.0);
        frame(5.0, 0.1);
        assert_eq!(frame(-5.0, 0.2), Matches { draws: 2, matched: 2, moved: 1, jumped: 0, jumping: 1 });
        let matches = frame(5.0, 0.3);
        assert!(matches.smooth() && matches.jumping == 1);
        assert_eq!(frame(-5.0, 9.0).jumped, 1, "the model jumped after moving");
        // an animation stepping every other frame
        let mut interpolation = Interpolation::default();
        let jumps: Vec<(u32, u32)> = [0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0]
            .into_iter()
            .map(|x| {
                if let Some(between) = interpolation.draw(&registers(3), &unit(x), None, true) {
                    interpolation.give_back(between);
                }
                let matches = interpolation.frame_done();
                (matches.jumped, matches.jumping)
            })
            .collect();
        assert_eq!(jumps, [(0, 0), (0, 0), (1, 0), (0, 0), (0, 1), (0, 0), (0, 1)]);
    }

    /// a model drawn twice, the two eyes of the 3D, in the other order the
    /// next frame, matches the draw nearest it, not the one in its place.
    #[test]
    fn draws_drawn_the_other_way_round_match_the_nearest() {
        let mut interpolation = Interpolation::default();
        interpolation.draw(&registers(3), &unit(-3.7), None, true);
        interpolation.draw(&registers(3), &unit(3.7), None, true);
        interpolation.frame_done();
        let right = interpolation.draw(&registers(3), &unit(3.8), None, true).expect("the right eye");
        assert!((right[0][3] - 3.75).abs() < 1e-6);
        interpolation.give_back(right);
        let left = interpolation.draw(&registers(3), &unit(-3.6), None, true).expect("the left eye");
        assert!((left[0][3] + 3.65).abs() < 1e-6);
        interpolation.give_back(left);
        assert_eq!(interpolation.frame_done(), Matches { draws: 2, matched: 2, moved: 2, jumped: 0, jumping: 0 });
    }

    /// a draw whose uniforms jump, as a matrix does when the camera cuts or
    /// a flag kept in a float flips, is drawn as it is, and a frame where
    /// most of them jump is not smooth.
    #[test]
    fn jumps_are_not_drawn_in_between() {
        let mut interpolation = Interpolation::default();
        interpolation.draw(&registers(3), &unit(100.0), None, true);
        interpolation.draw(&registers(4), &unit(0.0), None, true);
        interpolation.frame_done();
        let between = interpolation.draw(&registers(3), &unit(120.0), None, true).expect("a fifth of its distance");
        interpolation.give_back(between);
        assert!(interpolation.draw(&registers(4), &unit(1.0), None, true).is_none(), "a flag flipped");
        let matches = interpolation.frame_done();
        assert_eq!(matches, Matches { draws: 2, matched: 2, moved: 1, jumped: 1, jumping: 0 });
        assert!(!matches.smooth());
        interpolation.draw(&registers(3), &unit(200.0), None, true);
        assert_eq!(interpolation.frame_done().jumped, 1, "five sixths of its distance");
    }

    /// a draw through a geometry shader is drawn as it is, and never
    /// matches one without it.
    #[test]
    fn geometry_shaders_are_drawn_as_they_are() {
        let mut interpolation = Interpolation::default();
        let geometry = unit(0.0);
        interpolation.draw(&registers(3), &unit(0.0), Some(&geometry), true);
        interpolation.frame_done();
        assert!(interpolation.draw(&registers(3), &unit(0.2), None, true).is_none(), "no geometry stage");
        assert!(interpolation.draw(&registers(3), &unit(0.2), Some(&geometry), true).is_none());
        assert_eq!(interpolation.frame_done(), Matches { draws: 2, matched: 1, moved: 0, jumped: 0, jumping: 0 });
    }

    /// only the uniforms that place the vertices move in between, a
    /// texture's offset into a sheet of sprites stays as the frame has it,
    /// and its jumping does not count.
    #[test]
    fn only_what_places_the_vertices_moves() {
        // o0 = c0 . v0, o2 = c1, with o0 the position and o2 a texture's
        // coordinates
        let mut program = ShaderUnit::new();
        program.program[0] = (0x02 << 26) | (0x20 << 12);
        program.program[1] = (0x13 << 26) | (0x02 << 21) | (0x21 << 12);
        program.program[2] = 0x22 << 26;
        program.descriptors[0] = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        program.prepare();
        let mut registers = registers(3);
        registers[REG_SHADER_OUTPUT_TOTAL] = 3;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 2] = 0x1F1F_0D0C;
        let at = |x: f32, sprite: f32| {
            let mut unit = program.clone();
            unit.float_uniforms[0] = [1.0, 0.0, 0.0, x];
            unit.float_uniforms[1] = [sprite, 0.0, 0.0, 0.0];
            unit
        };
        let mut interpolation = Interpolation::default();
        interpolation.draw(&registers, &at(0.0, 0.25), None, true);
        interpolation.frame_done();
        let between = interpolation.draw(&registers, &at(0.4, 0.5), None, true).expect("moved");
        assert_eq!(between[0], [1.0, 0.0, 0.0, 0.2]);
        assert_eq!(between[1], [0.5, 0.0, 0.0, 0.0], "the next sprite, not half of each");
        interpolation.give_back(between);
        interpolation.frame_done();
        assert!(interpolation.draw(&registers, &at(0.4, 0.75), None, true).is_none(), "only the sprite changed");
        interpolation.frame_done();
        assert!(interpolation.draw(&registers, &at(0.6, 9.0), None, true).is_some(), "the sprite's jump does not count");
    }

    /// uniforms the coordinates may read through an address register too,
    /// a block that holds both a sprite's place and its offset into its
    /// sheet, move but for steps between round values, the offsets'.
    #[test]
    fn shared_uniforms_move_but_for_round_steps() {
        // o0 = c[a0.x + 20] . v0, o2 = c[a0.x + 20]
        let mut program = ShaderUnit::new();
        program.program[0] = (0x02 << 26) | (1 << 19) | ((0x20 + 20) << 12);
        program.program[1] = (0x13 << 26) | (0x02 << 21) | (1 << 19) | ((0x20 + 20) << 12);
        program.program[2] = 0x22 << 26;
        program.descriptors[0] = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        program.prepare();
        let mut registers = registers(3);
        registers[REG_SHADER_OUTPUT_TOTAL] = 3;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 2] = 0x1F1F_0D0C;
        let at = |x: f32| {
            let mut unit = program.clone();
            unit.float_uniforms[20] = [1.0, 0.0, 0.0, x];
            unit
        };
        let mut interpolation = Interpolation::default();
        interpolation.draw(&registers, &at(0.25), None, true);
        interpolation.frame_done();
        assert!(interpolation.draw(&registers, &at(0.5), None, true).is_none(), "a step between round values");
        interpolation.frame_done();
        interpolation.draw(&registers, &at(0.3), None, true);
        interpolation.frame_done();
        let between = interpolation.draw(&registers, &at(0.4), None, true).expect("a move");
        assert!((between[20][3] - 0.35).abs() < 1e-6);
    }
}
