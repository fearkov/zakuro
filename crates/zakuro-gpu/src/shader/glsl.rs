//! a vertex shader program turned into GLSL ahead of time, doing operation
//! for operation what shade.vert does interpreting it, so the GPU runs the
//! title's instructions instead of an interpreter. everything the program
//! fixes is worked out here, the instructions, their swizzles and masks,
//! where each input sits and where each output goes, and only what changes
//! from draw to draw, the uniforms and the viewport, is read as it runs.
//!
//! a program without flow instructions becomes straight code. one with
//! them becomes runs of instructions between the addresses execution can
//! come in at, with shade.vert's block stack between them, so calls, ifs,
//! loops, jumps and breaks behave exactly as interpreted, odd ones too.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write;
use std::sync::Arc;

use super::isa::OpCode;
use super::{Op, Operand, Program, ShaderUnit, PROGRAM_SIZE};

/// the semantics shade.vert's main reads, one per output component.
pub(crate) const SEMANTICS: usize = 24;

/// executed instructions after which shade.vert gives up on a program. the
/// translation counts a run's at its start, and stops a run that crosses
/// the limit before it writes an output shade.vert would not have, what it
/// ran past the limit until then only touching what no one reads after.
const BUDGET: u32 = 0x10000;

/// the address an end sends execution to, past any block's end.
const FINISHED: u32 = u32::MAX;

/// reachable instructions past which a program stays interpreted. real
/// ones are far shorter, but execution falling into the words after a
/// program runs to the end of the program memory, and a shader that long
/// takes a long while to compile.
const LONGEST: usize = 1024;

/// the blocks shade.vert's stack holds, pushes past them are dropped.
const BLOCK_LIMIT: usize = 16;

/// the states, where execution is and the blocks open, the translation
/// follows a program through before it leaves it to the interpreter.
const STATES: usize = 1 << 16;

impl ShaderUnit {
    /// the program from its entry point translated, as translate does.
    #[cfg(test)]
    pub(crate) fn translate(&self, semantics: &[u32; SEMANTICS]) -> Result<String, String> {
        let program = self.prepared().ok_or("the program was not prepared")?;
        translate(&program, self.entry_point, semantics)
    }

    /// the program as prepare decoded it, to translate it elsewhere.
    pub(crate) fn prepared(&self) -> Option<Arc<Program>> {
        self.decoded.clone()
    }
}

/// a program from an entry point as a GLSL vertex shader with shade.vert's
/// interface, its outputs going where semantics says, the output register
/// times four plus the component for each, or MISSING or ZERO as
/// hardware.rs has them.
pub(crate) fn translate(program: &Program, entry_point: u32, semantics: &[u32; SEMANTICS]) -> Result<String, String> {
    let translator = Translator::new(program, entry_point)?;
    let length: usize = translator.runs.values().map(|run| run.length() as usize).sum();
    if length > LONGEST {
        return Err(format!("{length} reachable instructions, more than {LONGEST}"));
    }
    Ok(translator.shader(semantics))
}

/// a vertex shader in GLSL to SPIR-V for vkCreateShaderModule, or why not.
/// naga can panic on some inputs, so a panic is an error too, and the caller
/// keeps the interpreter to fall back on.
pub(crate) fn compile(source: &str) -> Result<Vec<u32>, String> {
    use naga::back::spv;
    use naga::front::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let run = || -> Result<Vec<u32>, String> {
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(naga::ShaderStage::Vertex), source)
            .map_err(|errors| errors.emit_to_string_with_path(source, "translated vertex shader"))?;
        let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .map_err(|error| error.emit_to_string_with_path(source, "translated vertex shader"))?;
        let options = spv::Options {
            // no newer, naga's private arrays carry a stride later versions
            // reject
            lang_version: (1, 0),
            // the default flags turn y around, as WebGPU wants, and label
            // everything
            flags: spv::WriterFlags::empty(),
            // the default counts every loop's iterations
            force_loop_bounding: false,
            ..spv::Options::default()
        };
        let pipeline = spv::PipelineOptions { shader_stage: naga::ShaderStage::Vertex, entry_point: "main".into() };
        spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|error| error.to_string())
    };
    std::panic::catch_unwind(run).unwrap_or_else(|_| Err("naga panicked".to_owned()))
}

/// instructions from one address up to the next place execution can come
/// in at or go elsewhere.
struct Run {
    /// the arithmetic instructions, by address.
    body: Vec<u32>,
    exit: Exit,
}

#[derive(Clone, Copy)]
enum Exit {
    /// on to the next address, maybe past the program.
    Fall(u32),
    End,
    /// the flow instruction at an address.
    Flow(u32),
}

impl Run {
    /// the instructions it executes, as shade.vert counts them.
    fn length(&self) -> u32 {
        self.body.len() as u32 + u32::from(!matches!(self.exit, Exit::Fall(_)))
    }
}

struct Translator<'a> {
    ops: &'a [Op],
    /// the input registers stage_shading packs, those any word reads.
    packed: u16,
    entry: u32,
    /// the runs execution can reach, by their first address.
    runs: BTreeMap<u32, Run>,
    /// the input registers the reachable code reads.
    read: u16,
}

impl<'a> Translator<'a> {
    fn new(program: &'a Program, entry: u32) -> Result<Translator<'a>, String> {
        let ops = &program.ops[..];
        // split where any word could send execution, find what is reachable,
        // then split again only where the reachable flow sends it, which
        // words of earlier programs left in memory no longer cut up
        let everywhere = starts(ops, entry, 0..ops.len() as u32);
        let first = reachable(ops, &everywhere, entry)?;
        let flows: Vec<u32> = first
            .values()
            .filter_map(|run| match run.exit {
                Exit::Flow(at) => Some(at),
                _ => None,
            })
            .collect();
        let runs = reachable(ops, &starts(ops, entry, flows), entry)?;
        Ok(Translator { ops, packed: program.inputs, entry, runs, read: 0 })
    }

    /// whether anything reachable is a flow instruction, which needs the
    /// block stack, otherwise the program runs straight to its end.
    fn flows(&self) -> bool {
        self.runs.values().any(|run| matches!(run.exit, Exit::Flow(_)))
    }

    fn shader(mut self, semantics: &[u32; SEMANTICS]) -> String {
        let flows = self.flows();
        let mut run = String::new();
        if flows {
            self.dispatch(&mut run);
        } else {
            self.straight(&mut run);
        }

        let mut out = String::with_capacity(run.len() + 8192);
        out.push_str(HEADER);
        if flows {
            out.push_str(BLOCKS);
        }
        out.push_str("void run() {\n");
        // the inputs the code reads, each once, where stage_shading packed
        // them, a vertex's read registers in order
        let stride = self.packed.count_ones();
        if self.read != 0 {
            let _ = writeln!(out, "    uint base = uint(gl_VertexIndex) * {stride}u;");
        }
        for register in (0..16).filter(|register| self.read & (1 << register) != 0) {
            let slot = (self.packed & ((1u16 << register) - 1)).count_ones();
            let _ = writeln!(out, "    vec4 v{register} = inputs[base + {slot}u];");
        }
        out.push_str(&run);
        out.push_str("}\n\n");

        out.push_str("void main() {\n");
        out.push_str("    for (int i = 0; i < 16; i++) {\n");
        out.push_str("        temps[i] = vec4(0.0);\n");
        out.push_str("        outputs[i] = vec4(0.0);\n");
        out.push_str("    }\n");
        out.push_str("    address = ivec3(0);\n");
        out.push_str("    condition = bvec2(false);\n");
        if flows {
            out.push_str("    blocks = 0;\n");
        }
        out.push_str("    run();\n");
        let component = |which: usize, missing: &str| semantic(semantics[which], missing);
        let vector = |range: std::ops::Range<usize>, missing: &str| {
            range.map(|which| component(which, missing)).collect::<Vec<_>>().join(", ")
        };
        let _ = writeln!(out, "    vec4 position = vec4({});", vector(0..4, "0.0"));
        out.push_str(EPILOGUE);
        let _ = writeln!(out, "    out_color = vec4({});", vector(4..8, "1.0"));
        let _ = writeln!(out, "    out_texcoords01 = vec4({});", vector(8..12, "0.0"));
        let _ = writeln!(out, "    out_texcoord2 = vec2({});", vector(12..14, "0.0"));
        out.push_str("    out_depth = position.z / position.w * depth_map.x + depth_map.y;\n");
        let _ = writeln!(out, "    out_quaternion = vec4({});", vector(14..18, "0.0"));
        let _ = writeln!(out, "    out_view = vec3({});", vector(18..21, "0.0"));
        out.push_str("}\n");
        out
    }

    /// a program without flow, its runs one after another from the entry.
    fn straight(&mut self, out: &mut String) {
        let mut at = self.entry;
        while let Some(run) = self.runs.get(&at) {
            let (body, exit) = (run.body.clone(), run.exit);
            for address in body {
                self.instruction(out, address, "    ");
            }
            match exit {
                Exit::Fall(next) => at = next,
                _ => break,
            }
        }
    }

    /// a program with flow, its runs picked by address, the block stack
    /// handled between them as shade.vert handles it between instructions.
    /// every address a block can end at starts a run, so no end goes by
    /// unseen.
    fn dispatch(&mut self, out: &mut String) {
        let _ = writeln!(out, "    uint pc = {}u;", self.entry);
        out.push_str("    uint budget = 0u;\n");
        out.push_str("    while (true) {\n");
        out.push_str(BLOCK_ENDS);
        let _ = writeln!(out, "        if (pc >= {PROGRAM_SIZE}u || budget >= {BUDGET}u) {{");
        out.push_str("            return;\n");
        out.push_str("        }\n");
        out.push_str("        switch (pc) {\n");
        let starts: Vec<u32> = self.runs.keys().copied().collect();
        for start in starts {
            let run = &self.runs[&start];
            let (length, body, exit) = (run.length(), run.body.clone(), run.exit);
            let _ = writeln!(out, "        case {start}u:");
            let _ = writeln!(out, "            budget += {length}u;");
            for (i, address) in (0u32..).zip(body) {
                // shade.vert runs instruction i of the run only while the
                // budget before the run plus i is under the limit
                let op = &self.ops[address as usize];
                if i > 0 && op.opcode.writes() && op.destination < 0x10 {
                    let _ = writeln!(out, "            if (budget >= {}u) {{", BUDGET + length - i);
                    out.push_str("                return;\n");
                    out.push_str("            }\n");
                }
                self.instruction(out, address, "            ");
            }
            match exit {
                Exit::Fall(next) => {
                    let _ = writeln!(out, "            pc = {next}u;");
                }
                Exit::End => {
                    let _ = writeln!(out, "            pc = {FINISHED}u;");
                }
                Exit::Flow(at) => self.flow(out, at),
            }
            out.push_str("            break;\n");
        }
        out.push_str("        default:\n");
        let _ = writeln!(out, "            pc = {FINISHED}u;");
        out.push_str("            break;\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    /// a flow instruction, as shade.vert's run does it.
    fn flow(&self, out: &mut String, at: u32) {
        let op = &self.ops[at as usize];
        let (destination, count) = op.instruction.flow_target();
        let next = at + 1;
        let taken = taken(op);
        let indent = "            ";
        match op.opcode {
            OpCode::Call | OpCode::CallC | OpCode::CallU => {
                let _ = writeln!(out, "{indent}if ({taken}) {{");
                let _ = writeln!(out, "{indent}    pc = enter({destination}u, {count}u, {next}u);");
                let _ = writeln!(out, "{indent}}} else {{");
                let _ = writeln!(out, "{indent}    pc = {next}u;");
                let _ = writeln!(out, "{indent}}}");
            }
            OpCode::JmpC | OpCode::JmpU => {
                let _ = writeln!(out, "{indent}pc = {taken} ? {destination}u : {next}u;");
            }
            OpCode::IfU | OpCode::IfC => {
                // the body, then on past the else block, or the else block
                let body = destination - next.min(destination);
                let after = destination + count;
                let _ = writeln!(out, "{indent}if ({taken}) {{");
                let _ = writeln!(out, "{indent}    pc = enter({next}u, {body}u, {after}u);");
                let _ = writeln!(out, "{indent}}} else {{");
                let _ = writeln!(out, "{indent}    pc = enter({destination}u, {count}u, {after}u);");
                let _ = writeln!(out, "{indent}}}");
            }
            OpCode::Loop => {
                let integer = op.instruction.integer_index();
                let end = destination + 1;
                let _ = writeln!(out, "{indent}{{");
                let _ = writeln!(out, "{indent}    ivec4 integer = integers[{integer}];");
                let _ = writeln!(out, "{indent}    address.z = integer.y;");
                let _ = writeln!(out, "{indent}    if (blocks < 16) {{");
                let _ = writeln!(out, "{indent}        block_end[blocks] = {end}u;");
                let _ = writeln!(out, "{indent}        block_return[blocks] = {end}u;");
                let _ = writeln!(out, "{indent}        block_repeat[blocks] = uint(integer.x);");
                let _ = writeln!(out, "{indent}        block_increment[blocks] = integer.z;");
                let _ = writeln!(out, "{indent}        block_start[blocks] = {next}u;");
                let _ = writeln!(out, "{indent}        block_loop[blocks] = true;");
                let _ = writeln!(out, "{indent}        blocks++;");
                let _ = writeln!(out, "{indent}    }}");
                let _ = writeln!(out, "{indent}    pc = {next}u;");
                let _ = writeln!(out, "{indent}}}");
            }
            OpCode::Break | OpCode::BreakC => {
                // out of the innermost loop and anything open inside it
                let _ = writeln!(out, "{indent}{{");
                let _ = writeln!(out, "{indent}    uint next = {next}u;");
                let _ = writeln!(out, "{indent}    if ({taken}) {{");
                let _ = writeln!(out, "{indent}        while (blocks > 0) {{");
                let _ = writeln!(out, "{indent}            blocks--;");
                let _ = writeln!(out, "{indent}            if (block_loop[blocks]) {{");
                let _ = writeln!(out, "{indent}                next = block_return[blocks];");
                let _ = writeln!(out, "{indent}                break;");
                let _ = writeln!(out, "{indent}            }}");
                let _ = writeln!(out, "{indent}        }}");
                let _ = writeln!(out, "{indent}    }}");
                let _ = writeln!(out, "{indent}    pc = next;");
                let _ = writeln!(out, "{indent}}}");
            }
            _ => unreachable!("only flow instructions end a run with flow"),
        }
    }

    /// an arithmetic instruction, as shade.vert's arithmetic and
    /// multiply_add do it, all its sources read before anything is written.
    fn instruction(&mut self, out: &mut String, at: u32, indent: &str) {
        use OpCode::*;
        let op = self.ops[at as usize];
        let sources = match op.opcode {
            Mad | MadI => 3,
            Add | Dp3 | Dp4 | Dph | DphI | Dst | DstI | Mul | Sge | SgeI | Slt | SltI | Max | Min | Cmp => 2,
            Ex2 | Lg2 | LitP | Flr | Rcp | Rsq | Mov | Mova => 1,
            // nop, emits and the encodings that do nothing
            _ => return,
        };
        let mask = mask(op.mask);
        let effect = match op.opcode {
            Mova => op.mask & 0b1100 != 0,
            Cmp | LitP => true,
            _ => !mask.is_empty(),
        };
        if !effect {
            return;
        }
        let _ = writeln!(out, "{indent}{{");
        for (name, source) in ["a", "b", "c"].iter().zip(&op.sources[..sources]) {
            let value = self.operand(source);
            let _ = writeln!(out, "{indent}    vec4 {name} = {value};");
        }
        let result = match op.opcode {
            Add => "a + b".to_owned(),
            Dp3 => "vec4(dot3(a, b))".to_owned(),
            Dp4 => "vec4(dot4(a, b))".to_owned(),
            // the first source's w taken as one
            Dph | DphI => "vec4(dot3(a, b) + b.w)".to_owned(),
            Dst | DstI => "vec4(1.0, multiply(a.y, b.y), a.z, b.w)".to_owned(),
            Ex2 => "vec4(exp2(a.x))".to_owned(),
            Lg2 => "vec4(log2(a.x))".to_owned(),
            LitP => {
                let _ = writeln!(out, "{indent}    condition = bvec2(a.x >= 0.0, a.w >= 0.0);");
                "vec4(max(a.x, 0.0), clamp(a.y, -127.9961, 127.9961), 0.0, max(a.w, 0.0))".to_owned()
            }
            Mul => "multiply4(a, b)".to_owned(),
            Sge | SgeI => "vec4(greaterThanEqual(a, b))".to_owned(),
            Slt | SltI => "vec4(lessThan(a, b))".to_owned(),
            Flr => "floor(a)".to_owned(),
            // NaN behaving as on hardware, max(0, NaN) is NaN but
            // max(NaN, 0) is 0
            Max => "vec4(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z, a.w > b.w ? a.w : b.w)"
                .to_owned(),
            Min => "vec4(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z, a.w < b.w ? a.w : b.w)"
                .to_owned(),
            Rcp => "vec4(1.0 / a.x)".to_owned(),
            Rsq => "vec4(1.0 / sqrt(a.x))".to_owned(),
            Mov => "a".to_owned(),
            Mad | MadI => "multiply4(a, b) + c".to_owned(),
            Mova => {
                if op.mask & 0b1000 != 0 {
                    let _ = writeln!(out, "{indent}    address.x = to_int(a.x);");
                }
                if op.mask & 0b0100 != 0 {
                    let _ = writeln!(out, "{indent}    address.y = to_int(a.y);");
                }
                let _ = writeln!(out, "{indent}}}");
                return;
            }
            Cmp => {
                let [x, y] = op.instruction.compare_modes();
                let (x, y) = (compare(x, "a.x", "b.x"), compare(y, "a.y", "b.y"));
                let _ = writeln!(out, "{indent}    condition = bvec2({x}, {y});");
                let _ = writeln!(out, "{indent}}}");
                return;
            }
            _ => unreachable!("every opcode with sources has a result"),
        };
        let destination = register(op.destination);
        if mask.len() == 4 {
            let _ = writeln!(out, "{indent}    {destination} = {result};");
        } else if !mask.is_empty() {
            // a component at a time, the others keep their values
            let _ = writeln!(out, "{indent}    vec4 result = {result};");
            for component in mask.chars() {
                let _ = writeln!(out, "{indent}    {destination}.{component} = result.{component};");
            }
        }
        let _ = writeln!(out, "{indent}}}");
    }

    /// a source's value with its swizzle and negation, as shade.vert's
    /// source and operand read it.
    fn operand(&mut self, operand: &Operand) -> String {
        let register = operand.register;
        let value = if register < 0x10 {
            if self.packed & (1 << register) == 0 {
                // nothing packed for it, shade.vert reads zero
                "vec4(0.0)".to_owned()
            } else {
                self.read |= 1 << register;
                format!("v{register}")
            }
        } else if register < 0x20 {
            format!("temps[{}]", register - 0x10)
        } else {
            let uniform = register - 0x20;
            match operand.index {
                0 => format!("floats[{uniform}]"),
                index => format!("uniform_at({uniform} + address.{})", ["x", "y", "z"][(index as usize - 1).min(2)]),
            }
        };
        let swizzled = if operand.swizzle == [0, 1, 2, 3] {
            value
        } else {
            let swizzle: String = operand.swizzle.iter().map(|&component| b"xyzw"[component as usize & 3] as char).collect();
            format!("{value}.{swizzle}")
        };
        if operand.negate {
            format!("-{swizzled}")
        } else {
            swizzled
        }
    }
}

/// every address execution can come in at, the entry and whatever the flow
/// instructions at some addresses can send it to, a block's end included,
/// so that runs split there.
fn starts(ops: &[Op], entry: u32, flows: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
    let mut starts = BTreeSet::from([entry]);
    for at in flows {
        let op = &ops[at as usize];
        let (destination, count) = op.instruction.flow_target();
        let next = at + 1;
        match op.opcode {
            OpCode::Call | OpCode::CallC | OpCode::CallU => starts.extend([destination, destination + count, next]),
            // the body's end is the destination, or the body's start when
            // the destination comes first
            OpCode::IfU | OpCode::IfC => starts.extend([next, destination, destination + count]),
            OpCode::JmpC | OpCode::JmpU => starts.extend([destination, next]),
            OpCode::Loop => starts.extend([next, destination + 1]),
            OpCode::Break | OpCode::BreakC | OpCode::End => {
                starts.insert(next);
            }
            _ => {}
        }
    }
    starts
}

/// the run from start, up to the next start, an end or a flow instruction.
fn scan(ops: &[Op], starts: &BTreeSet<u32>, start: u32) -> Run {
    let mut body = Vec::new();
    let mut at = start;
    let exit = loop {
        if at as usize >= PROGRAM_SIZE || (at != start && starts.contains(&at)) {
            break Exit::Fall(at);
        }
        match ops[at as usize].opcode {
            OpCode::End => break Exit::End,
            OpCode::Call
            | OpCode::CallC
            | OpCode::CallU
            | OpCode::IfU
            | OpCode::IfC
            | OpCode::JmpC
            | OpCode::JmpU
            | OpCode::Loop
            | OpCode::Break
            | OpCode::BreakC => break Exit::Flow(at),
            _ => body.push(at),
        }
        at += 1;
    };
    Run { body, exit }
}

/// a block open on the stack as the translation follows execution, all of
/// it but how many more times a loop goes round, which the uniforms decide.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Open {
    end: u32,
    /// where execution goes once the block is done.
    back: u32,
    /// where a loop goes round again from.
    start: u32,
    is_loop: bool,
}

/// the runs execution can reach from the entry, following it with the
/// blocks it has open, the way shade.vert does, every way a condition or
/// a loop's count could send it. a block's end is only an end with the
/// block open, so the code a subroutine runs into at its end is not run.
fn reachable(ops: &[Op], starts: &BTreeSet<u32>, entry: u32) -> Result<BTreeMap<u32, Run>, String> {
    let mut runs = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut pending = vec![(entry, Vec::new())];
    while let Some((at, open)) = pending.pop() {
        for (at, open) in settle(at, open) {
            if at as usize >= PROGRAM_SIZE || !seen.insert((at, open.clone())) {
                continue;
            }
            if seen.len() > STATES {
                return Err(format!("its flow goes more than {STATES} ways"));
            }
            let exit = runs.entry(at).or_insert_with(|| scan(ops, starts, at)).exit;
            match exit {
                Exit::Fall(next) => pending.push((next, open)),
                Exit::End => {}
                Exit::Flow(flow) => pending.extend(follow(&ops[flow as usize], flow, open)),
            }
        }
    }
    Ok(runs)
}

/// where execution goes once the blocks ending where it is are done,
/// innermost first, as the top of the dispatch loop works it out, a loop
/// going round again or not.
fn settle(at: u32, open: Vec<Open>) -> Vec<(u32, Vec<Open>)> {
    let mut settled = Vec::new();
    let mut seen = HashSet::new();
    let mut pending = vec![(at, open)];
    while let Some((at, mut open)) = pending.pop() {
        if !seen.insert((at, open.clone())) {
            continue;
        }
        match open.last().cloned() {
            Some(top) if top.end == at => {
                if top.is_loop {
                    pending.push((top.start, open.clone()));
                }
                open.pop();
                pending.push((top.back, open));
            }
            _ => settled.push((at, open)),
        }
    }
    settled
}

/// where a flow instruction can send execution and the blocks it leaves
/// open, both ways where its condition decides.
fn follow(op: &Op, at: u32, open: Vec<Open>) -> Vec<(u32, Vec<Open>)> {
    let (destination, count) = op.instruction.flow_target();
    let next = at + 1;
    let push = |mut open: Vec<Open>, block: Open| {
        if open.len() < BLOCK_LIMIT {
            open.push(block);
        }
        open
    };
    let call = Open { end: destination + count, back: next, start: destination, is_loop: false };
    match op.opcode {
        OpCode::Call => vec![(destination, push(open, call))],
        OpCode::CallC | OpCode::CallU => vec![(destination, push(open.clone(), call)), (next, open)],
        OpCode::IfU | OpCode::IfC => {
            let after = destination + count;
            let body = Open { end: next + (destination - next.min(destination)), back: after, start: next, is_loop: false };
            let otherwise = Open { end: after, back: after, start: destination, is_loop: false };
            vec![(next, push(open.clone(), body)), (destination, push(open, otherwise))]
        }
        OpCode::JmpC | OpCode::JmpU => vec![(destination, open.clone()), (next, open)],
        OpCode::Loop => {
            let block = Open { end: destination + 1, back: destination + 1, start: next, is_loop: true };
            vec![(next, push(open, block))]
        }
        OpCode::Break => vec![broken(open, next)],
        OpCode::BreakC => vec![broken(open.clone(), next), (next, open)],
        _ => Vec::new(),
    }
}

/// a break, out of the innermost loop and whatever is open inside it, or
/// on with nothing open when no loop is.
fn broken(mut open: Vec<Open>, next: u32) -> (u32, Vec<Open>) {
    while let Some(block) = open.pop() {
        if block.is_loop {
            return (block.back, open);
        }
    }
    (next, open)
}

/// whether a flow instruction is taken, as shade.vert's flow_condition
/// works it out, its fields folded in.
fn taken(op: &Op) -> String {
    let instruction = op.instruction;
    let set = format!("((bools & {}u) != 0u)", 1u32 << instruction.bool_index());
    match op.opcode {
        OpCode::Call | OpCode::Break => "true".to_owned(),
        OpCode::CallU | OpCode::IfU => set,
        // the low bit of its count turns it around
        OpCode::JmpU if instruction.0 & 1 == 0 => set,
        OpCode::JmpU => format!("!{set}"),
        _ => {
            let [x, y] = instruction.condition_reference();
            let x = if x { "condition.x" } else { "!condition.x" };
            let y = if y { "condition.y" } else { "!condition.y" };
            match instruction.condition_op() {
                0 => format!("({x} || {y})"),
                1 => format!("({x} && {y})"),
                2 => x.to_owned(),
                _ => y.to_owned(),
            }
        }
    }
}

/// a comparison as shade.vert's compare makes it. not equal is true for
/// NaN there, which only negating equal keeps whatever the compiler.
fn compare(mode: u32, a: &str, b: &str) -> String {
    match mode {
        0 => format!("{a} == {b}"),
        1 => format!("!({a} == {b})"),
        2 => format!("{a} < {b}"),
        3 => format!("{a} <= {b}"),
        4 => format!("{a} > {b}"),
        5 => format!("{a} >= {b}"),
        _ => "true".to_owned(),
    }
}

/// the components a write mask selects, its most significant bit x.
fn mask(mask: u32) -> String {
    [(8, 'x'), (4, 'y'), (2, 'z'), (1, 'w')]
        .iter()
        .filter(|&&(bit, _)| mask & bit != 0)
        .map(|&(_, component)| component)
        .collect()
}

/// an output or temporary register a destination names.
fn register(destination: u32) -> String {
    if destination < 0x10 {
        format!("outputs[{destination}]")
    } else {
        format!("temps[{}]", destination - 0x10)
    }
}

/// a semantic component's value, as shade.vert's semantic reads it, the
/// default for one no register carries, zero for one past the enabled
/// registers.
fn semantic(slot: u32, missing: &str) -> String {
    if slot == u32::MAX {
        missing.to_owned()
    } else if slot >= 64 {
        "0.0".to_owned()
    } else {
        format!("outputs[{}].{}", slot >> 2, ['x', 'y', 'z', 'w'][(slot & 3) as usize])
    }
}

/// shade.vert's interface and the helpers its arithmetic uses, the same
/// layouts so stage_shading fills it the same way, and no program buffer.
const HEADER: &str = r#"#version 450

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_texcoords01;
layout(location = 2) out vec2 out_texcoord2;
layout(location = 3) noperspective out float out_depth;
layout(location = 4) out vec4 out_quaternion;
layout(location = 5) out vec3 out_view;

layout(std430, set = 0, binding = 6) readonly buffer Shading {
    vec4 floats[96];
    ivec4 integers[4];
    uint bools;
    uint entry;
    uint stride;
    uint pad;
    uint slots[16];
    uint semantics[24];
    vec4 depth_map;
    vec4 viewport;
};

layout(std430, set = 0, binding = 7) readonly buffer Inputs {
    vec4 inputs[];
};

vec4 temps[16];
vec4 outputs[16];
ivec3 address;
bvec2 condition;

float multiply(float a, float b) {
    float product = a * b;
    return isnan(product) && !isnan(a) && !isnan(b) ? 0.0 : product;
}

int to_int(float value) {
    return isnan(value) ? 0 : int(clamp(value, -2147483648.0, 2147483520.0));
}

vec4 multiply4(vec4 a, vec4 b) {
    return vec4(multiply(a.x, b.x), multiply(a.y, b.y), multiply(a.z, b.z), multiply(a.w, b.w));
}

float dot3(vec4 a, vec4 b) {
    return multiply(a.x, b.x) + multiply(a.y, b.y) + multiply(a.z, b.z);
}

float dot4(vec4 a, vec4 b) {
    return dot3(a, b) + multiply(a.w, b.w);
}

vec4 uniform_at(int index) {
    return index >= 0 && index < 96 ? floats[index] : vec4(0.0);
}

"#;

/// shade.vert's block stack and enter, for programs with flow.
const BLOCKS: &str = r#"uint block_end[16];
uint block_return[16];
uint block_repeat[16];
int block_increment[16];
uint block_start[16];
bool block_loop[16];
int blocks;

uint enter(uint start, uint count, uint return_address) {
    if (blocks < 16) {
        block_end[blocks] = start + count;
        block_return[blocks] = return_address;
        block_repeat[blocks] = 0u;
        block_increment[blocks] = 0;
        block_start[blocks] = start;
        block_loop[blocks] = false;
        blocks++;
    }
    return start;
}

"#;

/// shade.vert's handling of the blocks that end where execution is, a
/// loop going round again, anything else returning, innermost first. naga
/// evaluates both sides of &&, so the stack is looked at only once it
/// holds something.
const BLOCK_ENDS: &str = r#"        while (blocks > 0) {
            int top = blocks - 1;
            if (pc != block_end[top]) {
                break;
            }
            address.z += block_increment[top];
            if (block_repeat[top] == 0u) {
                pc = block_return[top];
                blocks--;
            } else {
                block_repeat[top]--;
                pc = block_start[top];
            }
        }
"#;

/// shade.vert's main from the position on, the PICA placing vertices on a
/// sixteenth of a pixel, halves to even as both round them, and clipping z
/// to -w..0, which Vulkan has as 0..w.
const EPILOGUE: &str = r#"    if (position.w > 1e-5) {
        vec2 window = viewport.xy + (position.xy / position.w * 0.5 + 0.5) * viewport.zw;
        window = roundEven(window * 16.0) / 16.0;
        vec2 placed = ((window - viewport.xy) / viewport.zw * 2.0 - 1.0) * position.w;
        position = vec4(placed, position.z, position.w);
    }
    gl_Position = vec4(position.xy, -position.z, position.w);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// a little random number generator, the same numbers every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// an arithmetic instruction of any kind, its registers, swizzles,
    /// masks and address registers random.
    fn arithmetic(random: &mut Random) -> u32 {
        let ops = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13];
        let (destination, index, descriptor) = (random.below(32), random.below(4), random.below(128));
        let (wide, narrow) = (random.below(0x80), random.below(0x20));
        match random.below(10) {
            0..=4 => {
                let op = ops[random.below(ops.len() as u32) as usize];
                (op << 26) | (destination << 21) | (index << 19) | (wide << 12) | (narrow << 7) | descriptor
            }
            5 => {
                let op = [0x18, 0x19, 0x1A, 0x1B][random.below(4) as usize];
                (op << 26) | (destination << 21) | (index << 19) | (narrow << 14) | (wide << 7) | descriptor
            }
            6 => (0x2E << 26) | (random.below(8) << 24) | (random.below(8) << 21) | (index << 19) | (wide << 12) | (narrow << 7) | descriptor,
            7 => {
                let (src1, src3) = (random.below(0x20), random.below(0x20));
                (0b111 << 29) | (destination << 24) | (index << 22) | (src1 << 17) | (wide << 10) | (src3 << 5) | (descriptor & 0x1F)
            }
            8 => {
                let (src1, src2) = (random.below(0x20), random.below(0x20));
                (0b110 << 29) | (destination << 24) | (index << 22) | (src1 << 17) | (src2 << 12) | (wide << 5) | (descriptor & 0x1F)
            }
            // the encodings that do nothing, and a nop
            _ => [0x10 << 26, 0x1C << 26, 0x21 << 26][random.below(3) as usize] | random.below(1 << 20),
        }
    }

    /// a flow instruction of any kind, whatever it points at.
    fn flow(random: &mut Random, at: u32, length: u32) -> u32 {
        let destination = random.below(length + 1);
        let count = random.below(6);
        let condition = random.below(1 << 4) << 22;
        let target = (destination << 10) | count;
        match random.below(11) {
            0 => (0x24 << 26) | target,
            1 => (0x25 << 26) | condition | target,
            2 => (0x26 << 26) | condition | target,
            3 => (0x27 << 26) | condition | target,
            4 => (0x28 << 26) | condition | target,
            5 => (0x2C << 26) | condition | target,
            6 => (0x2D << 26) | condition | target,
            // a loop whose body is the next few instructions
            7 => (0x29 << 26) | (random.below(4) << 22) | ((at + 1 + random.below(4)) << 10),
            8 => 0x20 << 26,
            9 => (0x23 << 26) | condition,
            _ => 0x22 << 26,
        }
    }

    fn unit(program: &[u32], entry: u32, random: &mut Random) -> ShaderUnit {
        let mut unit = ShaderUnit::new();
        for (i, &word) in program.iter().enumerate() {
            unit.program[i] = word;
        }
        for descriptor in unit.descriptors.iter_mut() {
            *descriptor = random.next() & 0x7FFF_FFFF;
        }
        unit.entry_point = entry;
        unit.prepare();
        unit
    }

    /// semantics the way raster.rs makes them, some components on output
    /// registers, some missing and some past the enabled ones.
    fn semantics(random: &mut Random) -> [u32; SEMANTICS] {
        std::array::from_fn(|_| match random.below(8) {
            0 => u32::MAX,
            1 => u32::MAX - 1,
            _ => random.below(64),
        })
    }

    /// translates and compiles, failing with the source on any error, and
    /// checks the SPIR-V with spirv-val when it is installed.
    fn compiles(unit: &ShaderUnit, semantics: &[u32; SEMANTICS]) -> String {
        let glsl = unit.translate(semantics).unwrap();
        let words = compile(&glsl).unwrap_or_else(|error| panic!("{error}\n{glsl}"));
        static CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let check = CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("zakuro-translated-{}-{check}.spv", std::process::id()));
        std::fs::write(&path, words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        if let Ok(output) = std::process::Command::new("spirv-val").arg("--target-env").arg("vulkan1.3").arg(&path).output() {
            assert!(output.status.success(), "{}\n{glsl}", String::from_utf8_lossy(&output.stderr));
        }
        let _ = std::fs::remove_file(path);
        glsl
    }

    /// programs of nothing but arithmetic come out straight, with no
    /// dispatch, and compile.
    #[test]
    fn straight_programs_compile() {
        let mut random = Random(11);
        for _ in 0..150 {
            let length = 1 + random.below(40);
            let mut program: Vec<u32> = (0..length).map(|_| arithmetic(&mut random)).collect();
            program.push(0x22 << 26);
            let unit = unit(&program, 0, &mut random);
            let glsl = compiles(&unit, &semantics(&mut random));
            assert!(!glsl.contains("switch"), "{glsl}");
        }
    }

    /// programs with flow of every kind, wherever it points, compile, the
    /// odd ones too, jumps out of ifs, calls that never return, breaks
    /// with no loop and runs past the end of the program.
    #[test]
    fn programs_with_flow_compile() {
        let mut random = Random(12);
        let mut compiled = 0;
        for _ in 0..250 {
            let length = 4 + random.below(40);
            let mut program = Vec::new();
            for at in 0..length {
                program.push(if random.below(4) == 0 { flow(&mut random, at, length) } else { arithmetic(&mut random) });
            }
            program.push(0x22 << 26);
            let entry = random.below(4);
            let unit = unit(&program, entry, &mut random);
            // falling into the words after the program leaves it to the
            // interpreter
            let semantics = semantics(&mut random);
            if unit.translate(&semantics).is_ok() {
                compiles(&unit, &semantics);
                compiled += 1;
            }
        }
        assert!(compiled > 150, "only {compiled} of 250 programs translated");
    }

    /// execution falling into the words after a program, which run to the
    /// end of the program memory, leaves the program to the interpreter.
    #[test]
    fn programs_running_into_the_rest_of_memory_stay_interpreted() {
        let mut random = Random(14);
        // a jump past the end into words of zero, each an add
        let program = [(0x2C << 26) | (1 << 22) | (8 << 10), 0x22 << 26];
        let unit = unit(&program, 0, &mut random);
        assert!(unit.translate(&semantics(&mut random)).is_err());
    }

    /// a program running into the last word and past it, inside a call
    /// whose end is just past the program, returns from the call there,
    /// as the interpreter does.
    #[test]
    fn a_call_can_end_past_the_program() {
        let mut random = Random(13);
        let mut program = vec![0u32; PROGRAM_SIZE];
        // call 4090 for 6 instructions, the end at 4096, then end
        program[0] = (0x24 << 26) | (4090 << 10) | 6;
        program[1] = 0x22 << 26;
        for word in &mut program[4090..] {
            *word = arithmetic(&mut random) & !(0x3F << 26) | (0x13 << 26);
        }
        let unit = unit(&program, 0, &mut random);
        let glsl = compiles(&unit, &semantics(&mut random));
        assert!(glsl.contains("pc = 4096u;"), "{glsl}");
    }
}
