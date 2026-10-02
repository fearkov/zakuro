#version 450

// the PICA200's vertex shader, run on the GPU. each vertex interprets the
// title's program the way the CPU's shader interpreter does, then hands the
// fragment stages what the rasterizer would have, clipping and perspective
// left to the GPU. the PICA clips z to -w..0, which the position turns
// into Vulkan's 0..w.

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_texcoords01;
layout(location = 2) out vec2 out_texcoord2;
layout(location = 3) noperspective out float out_depth;
layout(location = 4) out vec4 out_quaternion;
layout(location = 5) out vec3 out_view;

layout(std430, set = 0, binding = 5) readonly buffer Program {
    uint code[4096];
    uint descriptors[128];
};

layout(std430, set = 0, binding = 6) readonly buffer Shading {
    vec4 floats[96];
    // per integer uniform, the count, start and step of a loop
    ivec4 integers[4];
    uint bools;
    uint entry;
    // the vec4s each vertex takes in the inputs
    uint stride;
    uint pad;
    // the slot each input register reads in a vertex, or none
    uint slots[16];
    // per semantic component, the output register times four plus the
    // component, or none, position xyzw, color rgba, texture coordinates
    // 0, 1 and 2 uv, quaternion xyzw and view xyz
    uint semantics[24];
    // x, the depth map's scale, y, its offset
    vec4 depth_map;
    // left, bottom, width and height
    vec4 viewport;
};

layout(std430, set = 0, binding = 7) readonly buffer Inputs {
    vec4 inputs[];
};

const uint NONE = 0xFFFFFFFFu;
const int BLOCKS = 16;

vec4 temps[16];
vec4 outputs[16];
ivec3 address;
bvec2 condition;

// open calls, if bodies and loops, innermost last
uint block_end[BLOCKS];
uint block_return[BLOCKS];
uint block_repeat[BLOCKS];
int block_increment[BLOCKS];
uint block_start[BLOCKS];
bool block_loop[BLOCKS];
int blocks = 0;

// zero rather than NaN for zero times infinity
float multiply(float a, float b) {
    float product = a * b;
    return isnan(product) && !isnan(a) && !isnan(b) ? 0.0 : product;
}

// a float as Rust casts it, saturating, NaN being zero
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

vec4 input_register(uint register) {
    uint slot = slots[register];
    return slot == NONE ? vec4(0.0) : inputs[uint(gl_VertexIndex) * stride + slot];
}

// a register's value, the wide ones reaching the uniforms, offset by an
// address register when index says so
vec4 source(uint register, uint index) {
    if (register < 0x10u) {
        return input_register(register);
    }
    if (register < 0x20u) {
        return temps[register - 0x10u];
    }
    int uniform_index = int(register) - 0x20;
    if (index != 0u) {
        uniform_index += address[index - 1u];
    }
    return uniform_index >= 0 && uniform_index < 96 ? floats[uniform_index] : vec4(0.0);
}

// a source with its descriptor's swizzle and negation
vec4 operand(uint descriptor, uint which, uint register, uint index) {
    uint shift = which == 1u ? 5u : (which == 2u ? 14u : 23u);
    uint pattern = (descriptor >> shift) & 0xFFu;
    vec4 value = source(register, index);
    vec4 swizzled = vec4(
        value[(pattern >> 6u) & 3u],
        value[(pattern >> 4u) & 3u],
        value[(pattern >> 2u) & 3u],
        value[pattern & 3u]
    );
    return ((descriptor >> (shift - 1u)) & 1u) != 0u ? -swizzled : swizzled;
}

void write_masked(uint register, uint mask, vec4 value) {
    // the mask's most significant bit selects x
    for (uint component = 0u; component < 4u; component++) {
        if ((mask & (8u >> component)) == 0u) {
            continue;
        }
        if (register < 0x10u) {
            outputs[register][component] = value[component];
        } else if (register < 0x20u) {
            temps[register - 0x10u][component] = value[component];
        }
    }
}

bool compare(uint mode, float a, float b) {
    switch (mode) {
    case 0u: return a == b;
    case 1u: return a != b;
    case 2u: return a < b;
    case 3u: return a <= b;
    case 4u: return a > b;
    case 5u: return a >= b;
    default: return true;
    }
}

bool flow_condition(uint word, uint opcode) {
    if (opcode == 0x24u) {
        return true;
    }
    uint bool_index = (word >> 22u) & 0xFu;
    bool set = (bools & (1u << bool_index)) != 0u;
    // callu, ifu
    if (opcode == 0x26u || opcode == 0x27u) {
        return set;
    }
    // jmpu, the low bit of its count inverts it
    if (opcode == 0x2Du) {
        return set == ((word & 1u) == 0u);
    }
    bool x = condition.x == (((word >> 25u) & 1u) != 0u);
    bool y = condition.y == (((word >> 24u) & 1u) != 0u);
    switch ((word >> 22u) & 3u) {
    case 0u: return x || y;
    case 1u: return x && y;
    case 2u: return x;
    default: return y;
    }
}

uint enter(uint start, uint count, uint return_address) {
    if (blocks < BLOCKS) {
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

void arithmetic(uint word, uint opcode) {
    uint descriptor = descriptors[word & 0x7Fu];
    bool inverted = opcode >= 0x18u && opcode <= 0x1Bu;
    uint src1 = inverted ? (word >> 14u) & 0x1Fu : (word >> 12u) & 0x7Fu;
    uint src2 = inverted ? (word >> 7u) & 0x7Fu : (word >> 7u) & 0x1Fu;
    uint index = (word >> 19u) & 3u;
    vec4 a = operand(descriptor, 1u, src1, inverted ? 0u : index);
    vec4 b = operand(descriptor, 2u, src2, inverted ? index : 0u);
    uint mask = descriptor & 0xFu;
    uint destination = (word >> 21u) & 0x1Fu;
    vec4 result;
    switch (opcode) {
    case 0x00u: result = a + b; break;
    case 0x01u: result = vec4(dot3(a, b)); break;
    case 0x02u: result = vec4(dot4(a, b)); break;
    // dph and its inverted form take the first operand's w as one
    case 0x03u:
    case 0x18u: result = vec4(dot3(a, b) + b.w); break;
    // dst and its inverted form
    case 0x04u:
    case 0x19u: result = vec4(1.0, multiply(a.y, b.y), a.z, b.w); break;
    case 0x05u: result = vec4(exp2(a.x)); break;
    case 0x06u: result = vec4(log2(a.x)); break;
    case 0x07u:
        condition = bvec2(a.x >= 0.0, a.w >= 0.0);
        result = vec4(max(a.x, 0.0), clamp(a.y, -127.9961, 127.9961), 0.0, max(a.w, 0.0));
        break;
    case 0x08u: result = multiply4(a, b); break;
    // sge and slt, and their inverted forms
    case 0x09u:
    case 0x1Au: result = vec4(greaterThanEqual(a, b)); break;
    case 0x0Au:
    case 0x1Bu: result = vec4(lessThan(a, b)); break;
    case 0x0Bu: result = floor(a); break;
    // written so NaN behaves as on hardware, max(0, NaN) is NaN but
    // max(NaN, 0) is 0
    case 0x0Cu:
        result = vec4(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z, a.w > b.w ? a.w : b.w);
        break;
    case 0x0Du:
        result = vec4(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z, a.w < b.w ? a.w : b.w);
        break;
    case 0x0Eu: result = vec4(1.0 / a.x); break;
    case 0x0Fu: result = vec4(1.0 / sqrt(a.x)); break;
    case 0x12u:
        if ((mask & 8u) != 0u) {
            address.x = to_int(a.x);
        }
        if ((mask & 4u) != 0u) {
            address.y = to_int(a.y);
        }
        return;
    case 0x13u: result = a; break;
    case 0x2Eu:
    case 0x2Fu:
        condition = bvec2(compare((word >> 24u) & 7u, a.x, b.x), compare((word >> 21u) & 7u, a.y, b.y));
        return;
    default:
        return;
    }
    write_masked(destination, mask, result);
}

void multiply_add(uint word, uint opcode) {
    uint descriptor = descriptors[word & 0x1Fu];
    bool inverted = opcode < 0x38u;
    uint src1 = (word >> 17u) & 0x1Fu;
    uint src2 = inverted ? (word >> 12u) & 0x1Fu : (word >> 10u) & 0x7Fu;
    uint src3 = inverted ? (word >> 5u) & 0x7Fu : (word >> 5u) & 0x1Fu;
    uint index = (word >> 22u) & 3u;
    vec4 a = operand(descriptor, 1u, src1, 0u);
    vec4 b = operand(descriptor, 2u, src2, inverted ? 0u : index);
    vec4 c = operand(descriptor, 3u, src3, inverted ? index : 0u);
    write_masked((word >> 24u) & 0x1Fu, descriptor & 0xFu, multiply4(a, b) + c);
}

void run() {
    uint pc = entry;
    // a program that never ends is a decoding bug, cap it rather than hang
    for (uint budget = 0u; budget < 0x10000u; budget++) {
        // finishing a block, go round a loop again, or return to whatever
        // follows it, several blocks can end at the same address
        while (blocks > 0 && pc == block_end[blocks - 1]) {
            int top = blocks - 1;
            address.z += block_increment[top];
            if (block_repeat[top] == 0u) {
                pc = block_return[top];
                blocks--;
            } else {
                block_repeat[top]--;
                pc = block_start[top];
            }
        }
        if (pc >= 4096u) {
            return;
        }
        uint word = code[pc];
        uint opcode = word >> 26u;
        uint next = pc + 1u;
        uint destination = (word >> 10u) & 0xFFFu;
        uint count = word & 0xFFu;
        if (opcode == 0x22u) {
            return;
        } else if (opcode >= 0x30u) {
            multiply_add(word, opcode);
        } else if (opcode == 0x24u || opcode == 0x25u || opcode == 0x26u) {
            if (flow_condition(word, opcode)) {
                next = enter(destination, count, pc + 1u);
            }
        } else if (opcode == 0x2Cu || opcode == 0x2Du) {
            if (flow_condition(word, opcode)) {
                next = destination;
            }
        } else if (opcode == 0x27u || opcode == 0x28u) {
            if (flow_condition(word, opcode)) {
                // the body, then past the else block
                next = enter(pc + 1u, destination - min(pc + 1u, destination), destination + count);
            } else {
                next = enter(destination, count, destination + count);
            }
        } else if (opcode == 0x29u) {
            ivec4 integer = integers[(word >> 22u) & 3u];
            address.z = integer.y;
            if (blocks < BLOCKS) {
                block_end[blocks] = destination + 1u;
                block_return[blocks] = destination + 1u;
                block_repeat[blocks] = uint(integer.x);
                block_increment[blocks] = integer.z;
                block_start[blocks] = pc + 1u;
                block_loop[blocks] = true;
                blocks++;
            }
        } else if (opcode == 0x20u || opcode == 0x23u) {
            if (opcode == 0x20u || flow_condition(word, opcode)) {
                // leave the innermost loop, and any if or call inside it
                while (blocks > 0) {
                    blocks--;
                    if (block_loop[blocks]) {
                        next = block_return[blocks];
                        break;
                    }
                }
            }
        } else if (opcode == 0x21u || opcode == 0x2Au || opcode == 0x2Bu) {
            // nop, and the geometry shader's emits, which a vertex has none of
        } else {
            arithmetic(word, opcode);
        }
        pc = next;
    }
}

// a semantic component's value, the default for one no register carries,
// zero for one past the registers the output mask enables
float semantic(uint which, float missing) {
    uint slot = semantics[which];
    if (slot == NONE) {
        return missing;
    }
    return slot >= 64u ? 0.0 : outputs[slot >> 2u][slot & 3u];
}

void main() {
    for (int i = 0; i < 16; i++) {
        temps[i] = vec4(0.0);
        outputs[i] = vec4(0.0);
    }
    address = ivec3(0);
    condition = bvec2(false);
    run();

    vec4 position = vec4(semantic(0u, 0.0), semantic(1u, 0.0), semantic(2u, 0.0), semantic(3u, 0.0));
    // the PICA places vertices on a sixteenth of a pixel, halves to even
    // the way translated programs place them too
    if (position.w > 1e-5) {
        vec2 window = viewport.xy + (position.xy / position.w * 0.5 + 0.5) * viewport.zw;
        window = roundEven(window * 16.0) / 16.0;
        position.xy = ((window - viewport.xy) / viewport.zw * 2.0 - 1.0) * position.w;
    }
    gl_Position = vec4(position.xy, -position.z, position.w);
    // no color is white, an untextured draw comes out lit
    out_color = vec4(semantic(4u, 1.0), semantic(5u, 1.0), semantic(6u, 1.0), semantic(7u, 1.0));
    out_texcoords01 = vec4(semantic(8u, 0.0), semantic(9u, 0.0), semantic(10u, 0.0), semantic(11u, 0.0));
    out_texcoord2 = vec2(semantic(12u, 0.0), semantic(13u, 0.0));
    out_depth = position.z / position.w * depth_map.x + depth_map.y;
    out_quaternion = vec4(semantic(14u, 0.0), semantic(15u, 0.0), semantic(16u, 0.0), semantic(17u, 0.0));
    out_view = vec3(semantic(18u, 0.0), semantic(19u, 0.0), semantic(20u, 0.0));
}
