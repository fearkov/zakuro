#version 450

// the PICA200's fragment stages for one draw, the texture units, fragment
// lighting and the six combiners, then the alpha test. they follow the
// software rasterizer, which follows the hardware.

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec4 in_texcoords01;
layout(location = 2) in vec2 in_texcoord2;
layout(location = 3) noperspective in float in_depth;
layout(location = 4) in vec4 in_quaternion;
layout(location = 5) in vec3 in_view;

layout(location = 0) out vec4 out_color;

// what decides the shape of the fragment stages comes in as specialization
// constants, a pipeline for each combination, so the driver compiles away
// the combiners' switches and the stages a draw does not use. per combiner
// stage the source, operand, combiner and scale registers, then the buffer
// update, the texture units' configuration, whether lighting is on, the
// alpha test's switch and function, and the depth mode
layout(constant_id = 0) const uint SOURCE0 = 0u;
layout(constant_id = 1) const uint SOURCE1 = 0u;
layout(constant_id = 2) const uint SOURCE2 = 0u;
layout(constant_id = 3) const uint SOURCE3 = 0u;
layout(constant_id = 4) const uint SOURCE4 = 0u;
layout(constant_id = 5) const uint SOURCE5 = 0u;
layout(constant_id = 6) const uint OPERAND0 = 0u;
layout(constant_id = 7) const uint OPERAND1 = 0u;
layout(constant_id = 8) const uint OPERAND2 = 0u;
layout(constant_id = 9) const uint OPERAND3 = 0u;
layout(constant_id = 10) const uint OPERAND4 = 0u;
layout(constant_id = 11) const uint OPERAND5 = 0u;
layout(constant_id = 12) const uint COMBINER0 = 0u;
layout(constant_id = 13) const uint COMBINER1 = 0u;
layout(constant_id = 14) const uint COMBINER2 = 0u;
layout(constant_id = 15) const uint COMBINER3 = 0u;
layout(constant_id = 16) const uint COMBINER4 = 0u;
layout(constant_id = 17) const uint COMBINER5 = 0u;
layout(constant_id = 18) const uint SCALE0 = 0u;
layout(constant_id = 19) const uint SCALE1 = 0u;
layout(constant_id = 20) const uint SCALE2 = 0u;
layout(constant_id = 21) const uint SCALE3 = 0u;
layout(constant_id = 22) const uint SCALE4 = 0u;
layout(constant_id = 23) const uint SCALE5 = 0u;
layout(constant_id = 24) const uint UPDATE = 0u;
layout(constant_id = 25) const uint TEXTURE_CONFIG = 0u;
layout(constant_id = 26) const uint LIGHTING = 0u;
layout(constant_id = 27) const uint ALPHA_TEST = 0u;
layout(constant_id = 28) const uint DEPTH_MODE = 0u;
// set, a generic pipeline that reads the above from the draw's registers
// instead, to draw with while the one for its combination compiles. it
// keeps lighting and the procedural texture's switch, a lot of code it goes
// without when they are off. it masks the rest as hardware.rs masks the
// constants, so the two draw the same
layout(constant_id = 29) const uint DYNAMIC = 0u;

layout(set = 0, binding = 0) uniform sampler2D texture0;
layout(set = 0, binding = 1) uniform sampler2D texture1;
layout(set = 0, binding = 2) uniform sampler2D texture2;

struct Light {
    vec4 specular0;
    vec4 specular1;
    vec4 diffuse;
    vec4 ambient;
    vec4 position;
    vec4 direction;
    vec4 spot;
    // bias and scale into the distance table
    vec4 distance;
    // x, 1 directional, 2 two sided, 4 and 8 geometric factors, 16 distance
    // attenuation, 32 spotlight, 64 shadowed, y, which light it is
    uvec4 flags;
};

layout(std140, set = 0, binding = 3) uniform Draw {
    // per stage the source, operand, combiner, constant and scale registers
    uvec4 tev[12];
    // the buffer update and buffer color registers, the alpha test and the
    // texture unit configuration
    uvec4 misc;
    // per unit the configuration word and the border color
    uvec4 units[3];
    // x, 1 a w-buffer, 2 depth from the fragment's z, y, lighting is on,
    // z and w, the depth map's scale and offset
    uvec4 flags;
    // x, the configuration, y, how many lights, z, bump mapping, w, shadow
    uvec4 light_config;
    // x, fresnel into the primary alpha, y, into the secondary, z, clamp
    // highlights, w, 1 when the half vector is read, 2 the view vector
    uvec4 light_flags;
    // distribution 0 and 1, fresnel, reflection red, green and blue and the
    // spotlight, each whether it is read, its input, abs and scale
    uvec4 lookups[7];
    vec4 global_ambient;
    Light lights[8];
    // the procedural texture's registers, the configuration, the noise's
    // u and v and its frequencies, then its table's configuration and
    // offset
    uvec4 proctex[2];
};

// a combiner stage's source, operand, combiner and scale registers, the
// pipeline's constants or, generic, the draw's
uvec4 stage_registers(uint stage, uvec4 constants) {
    if (DYNAMIC == 0u) {
        return constants;
    }
    uvec4 registers = tev[stage * 2u];
    return uvec4(registers.x & 0x0FFF0FFFu, registers.y & 0x00777FFFu, registers.z & 0x000F000Fu, tev[stage * 2u + 1u].x & 0x00030003u);
}

layout(std430, set = 0, binding = 4) readonly buffer Tables {
    // 24 tables of 256 entries, a value and the step to the next, then the
    // procedural texture's noise, color map and alpha map, 128 entries
    // each, and its 256 colors and their steps, two pairs an entry
    vec2 tables[];
};

const uint PROCTEX_MAPS = 24u * 256u;
const uint PROCTEX_COLORS = PROCTEX_MAPS + 3u * 128u;
const uint PROCTEX_STEPS = PROCTEX_COLORS + 2u * 256u;

const uint DISTRIBUTION0 = 0u;
const uint DISTRIBUTION1 = 1u;
const uint FRESNEL = 3u;
const uint REFLECT_BLUE = 4u;
const uint REFLECT_GREEN = 5u;
const uint REFLECT_RED = 6u;
const uint SPOTLIGHT = 8u;
const uint DISTANCE = 16u;

vec4 unpack_color(uint value) {
    return unpackUnorm4x8(value);
}

// a texture unit's sample, with v = 0 at the bottom row as the PICA has it
// and the border color outside the texture where the unit clamps to it
vec4 sample_unit(uint unit, vec2 uv) {
    uint config = units[unit].x;
    uint wrap_t = (config >> 8) & 7u;
    uint wrap_s = (config >> 12) & 7u;
    // precise as the combiners are. the coordinate itself the driver may
    // still interpolate a hair apart from one shader to another
    precise vec2 st = vec2(uv.x, 1.0 - uv.y);
    bool border_s = (wrap_s & 3u) == 1u && (st.x < 0.0 || st.x >= 1.0);
    bool border_t = (wrap_t & 3u) == 1u && (st.y < 0.0 || st.y >= 1.0);
    if (border_s || border_t) {
        return unpack_color(units[unit].y);
    }
    if (unit == 0u) {
        return texture(texture0, st);
    } else if (unit == 1u) {
        return texture(texture1, st);
    }
    return texture(texture2, st);
}

precise float lookup(uint table, uint entry, float delta) {
    vec2 value = tables[table * 256u + entry];
    return value.x + value.y * delta;
}

// a procedural texture map read at a coordinate from 0 to 1
precise float proctex_lookup(uint map, float coordinate) {
    float at = coordinate * 128.0;
    uint entry = min(uint(max(at, 0.0)), 127u);
    vec2 value = tables[PROCTEX_MAPS + map * 128u + entry];
    return value.x + (at - float(entry)) * value.y;
}

vec4 proctex_color(uint entry, uint base) {
    return vec4(tables[base + entry * 2u], tables[base + entry * 2u + 1u]);
}

// a pseudo-random value from -1 to 1 for a point of the noise's grid
float proctex_random(uint x, uint y) {
    const uint rows[16] = uint[16](0u, 4u, 10u, 8u, 4u, 9u, 7u, 12u, 5u, 15u, 13u, 14u, 11u, 15u, 2u, 11u);
    const uint mixes[16] = uint[16](10u, 2u, 15u, 8u, 0u, 7u, 4u, 5u, 5u, 13u, 2u, 6u, 13u, 9u, 3u, 14u);
    uint u = (((x % 9u + 2u) * 3u) & 0xFu) ^ rows[(x / 9u) & 0xFu];
    uint v = (((y % 9u + 2u) * 3u) & 0xFu) ^ rows[(y / 9u) & 0xFu];
    if ((u & 3u) == 1u) {
        v += 4u;
    }
    v ^= (u & 1u) * 6u;
    v += 10u + u;
    v &= 0xFu;
    v ^= mixes[u];
    return -1.0 + float(v) * 2.0 / 15.0;
}

float proctex_shift(float other, uint mode, uint clamping) {
    float amount = clamping == 3u ? 1.0 : 0.5;
    int o = int(other);
    if (mode == 1u) {
        return amount * float((o / 2) % 2);
    } else if (mode == 2u) {
        return amount * float(((o + 1) / 2) % 2);
    }
    return 0.0;
}

precise float proctex_clamp(float c, uint mode) {
    switch (mode) {
        case 0u: return c > 1.0 ? 0.0 : c;
        case 1u: return min(c, 1.0);
        case 2u: return c - floor(c);
        case 3u: {
            int whole = int(c);
            float part = c - float(whole);
            return (whole % 2) == 0 ? part : 1.0 - part;
        }
        case 4u: return c <= 0.5 ? 0.0 : 1.0;
        default: return clamp(c, 0.0, 1.0);
    }
}

precise float proctex_combine(float u, float v, uint function) {
    float len = sqrt(u * u + v * v);
    switch (function) {
        case 0u: return u;
        case 1u: return u * u;
        case 2u: return v;
        case 3u: return v * v;
        case 4u: return (u + v) * 0.5;
        case 5u: return (u * u + v * v) * 0.5;
        case 6u: return min(len, 1.0);
        case 7u: return min(u, v);
        case 8u: return max(u, v);
        case 9u: return min(((u + v) * 0.5 + len) * 0.5, 1.0);
        default: return 0.0;
    }
}

// texture 3, made up from its coordinates, as the software rasterizer has
// it. precise as the combiners are, as is lighting
precise vec4 procedural(vec2 uv) {
    uint config = proctex[0].x;
    uvec2 clamps = uvec2(config & 7u, (config >> 3) & 7u);
    float u = abs(uv.x);
    float v = abs(uv.y);
    vec2 shifts = vec2(proctex_shift(v, (config >> 16) & 3u, clamps.x), proctex_shift(u, (config >> 18) & 3u, clamps.y));
    if ((config & (1u << 15)) != 0u) {
        uint noise_u = proctex[0].y;
        uint noise_v = proctex[0].z;
        vec2 frequency = unpackHalf2x16(proctex[0].w);
        float x = 9.0 * frequency.x * abs(u + float(noise_u >> 16) / 4096.0);
        float y = 9.0 * frequency.y * abs(v + float(noise_v >> 16) / 4096.0);
        uint xi = uint(x);
        uint yi = uint(y);
        float xf = x - float(xi);
        float yf = y - float(yi);
        float g0 = proctex_random(xi, yi) * (xf + yf);
        float g1 = proctex_random(xi + 1u, yi) * (xf + yf - 1.0);
        float g2 = proctex_random(xi, yi + 1u) * (xf + yf - 1.0);
        float g3 = proctex_random(xi + 1u, yi + 1u) * (xf + yf - 2.0);
        float s = proctex_lookup(0u, xf);
        float t = proctex_lookup(0u, yf);
        float noise = mix(mix(g0, g1, s), mix(g2, g3, s), t);
        u = abs(u + noise * float(int(noise_u << 16) >> 16) / 4095.0);
        v = abs(v + noise * float(int(noise_v << 16) >> 16) / 4095.0);
    }
    u = proctex_clamp(u + shifts.x, clamps.x);
    v = proctex_clamp(v + shifts.y, clamps.y);

    float coordinate = proctex_lookup(1u, proctex_combine(u, v, (config >> 6) & 0xFu));
    uint lut = proctex[1].x;
    float width = float((lut >> 11) & 0xFFu);
    float at = float(proctex[1].y & 0xFFu) + coordinate * max(width - 1.0, 0.0);
    uint filtering = lut & 7u;
    vec4 color;
    if (filtering == 1u || filtering == 3u || filtering == 5u) {
        uint entry = min(uint(max(at, 0.0)), 255u);
        color = proctex_color(entry, PROCTEX_COLORS) + (at - float(entry)) * proctex_color(entry, PROCTEX_STEPS);
    } else {
        color = proctex_color(min(uint(max(round(at), 0.0)), 255u), PROCTEX_COLORS);
    }
    color = clamp(color / 255.0, 0.0, 1.0);
    // alpha of its own skips the color table, the map gives it
    if ((config & (1u << 14)) != 0u) {
        color.a = clamp(proctex_lookup(2u, proctex_combine(u, v, (config >> 10) & 0xFu)), 0.0, 1.0);
    }
    return color;
}

precise vec3 rotate(vec4 q, vec3 v) {
    vec3 inner = cross(q.xyz, v);
    return v + 2.0 * cross(q.xyz, inner + v * q.w);
}

precise float quantize(float c) {
    return floor(clamp(c, 0.0, 1.0) * 255.0) / 255.0;
}

// what fragment lighting leaves in the primary and secondary colors
void shade(vec4 textures[4], out vec4 diffuse_out, out vec4 specular_out) {
    uint config = light_config.x;
    float length_q = length(in_quaternion);
    vec4 q = length_q > 0.0 ? in_quaternion / length_q : vec4(0.0, 0.0, 0.0, 1.0);

    uint shadow_word = light_config.w;
    vec4 shadow = vec4(1.0);
    if ((shadow_word & 1u) != 0u) {
        shadow = textures[(shadow_word >> 4) & 3u];
        if ((shadow_word & 0x100u) != 0u) {
            shadow = vec4(1.0) - shadow;
        }
    }

    vec3 surface_normal = vec3(0.0, 0.0, 1.0);
    vec3 surface_tangent = vec3(1.0, 0.0, 0.0);
    uint bump = light_config.z;
    uint bump_mode = bump & 3u;
    if (bump_mode == 1u) {
        surface_normal = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
        if ((bump & 0x100u) != 0u) {
            surface_normal.z = sqrt(max(1.0 - dot(surface_normal.xy, surface_normal.xy), 0.0));
        }
    } else if (bump_mode == 2u) {
        surface_tangent = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
    }
    vec3 normal = rotate(q, surface_normal);
    vec3 tangent = config == 8u ? rotate(q, surface_tangent) : surface_tangent;
    bool needs_half = (light_flags.w & 1u) != 0u;
    bool needs_view = (light_flags.w & 2u) != 0u;
    vec3 view = in_view;
    vec3 norm_view = needs_view ? normalize(view) : vec3(0.0);

    precise vec4 diffuse_sum = vec4(0.0, 0.0, 0.0, 1.0);
    precise vec4 specular_sum = vec4(0.0, 0.0, 0.0, 1.0);
    uint count = light_config.y;
    for (uint slot = 0u; slot < count; slot++) {
        Light light = lights[slot];
        uint flags = light.flags.x;
        uint number = light.flags.y;
        bool two_sided = (flags & 2u) != 0u;
        vec3 light_vector = (flags & 1u) != 0u ? light.direction.xyz : normalize(light.position.xyz + view);
        vec3 half_vector = vec3(0.0);
        vec3 half_unit = vec3(0.0);
        if (needs_half) {
            half_vector = norm_view + light_vector;
            half_unit = normalize(half_vector);
        }

        float distance = 1.0;
        if ((flags & 16u) != 0u) {
            vec3 offset = -view - light.position.xyz;
            float place = clamp(light.distance.y * length(offset) + light.distance.x, 0.0, 1.0);
            float entry = clamp(floor(place * 256.0), 0.0, 255.0);
            distance = lookup(DISTANCE + number, uint(entry), place * 256.0 - entry);
        }

        float values[7];
        for (uint i = 0u; i < 7u; i++) {
            values[i] = 1.0;
            uvec4 l = lookups[i];
            if (l.x == 0u || (i == 6u && (flags & 32u) == 0u)) {
                continue;
            }
            float result = 0.0;
            switch (l.y) {
                case 0u: result = dot(normal, half_unit); break;
                case 1u: result = dot(norm_view, half_unit); break;
                case 2u: result = dot(normal, norm_view); break;
                case 3u: result = dot(light_vector, normal); break;
                case 4u: result = dot(light_vector, light.spot.xyz); break;
                case 5u:
                    if (config == 8u) {
                        float along = dot(normal, half_unit);
                        result = dot(half_unit - normal * along, tangent);
                    }
                    break;
                default: break;
            }
            uint entry;
            float delta;
            if (l.z != 0u) {
                result = two_sided ? abs(result) : max(result, 0.0);
                float e = clamp(floor(result * 256.0), 0.0, 255.0);
                entry = uint(e);
                delta = result * 256.0 - e;
            } else {
                // the signed index wraps into the upper half of the table
                float e = clamp(floor(result * 128.0), -128.0, 127.0);
                entry = uint(int(e)) & 0xFFu;
                delta = result * 128.0 - e;
            }
            uint table = i == 0u ? DISTRIBUTION0
                : i == 1u ? DISTRIBUTION1
                : i == 2u ? FRESNEL
                : i == 3u ? REFLECT_RED
                : i == 4u ? REFLECT_GREEN
                : i == 5u ? REFLECT_BLUE
                : SPOTLIGHT + number;
            values[i] = uintBitsToFloat(l.w) * lookup(table, entry, delta);
        }

        float spot = values[6];
        float red = values[3];
        vec3 reflect_color = vec3(red, lookups[4].x != 0u ? values[4] : red, lookups[5].x != 0u ? values[5] : red);
        vec3 specular0 = light.specular0.rgb * values[0];
        vec3 specular1 = values[1] * reflect_color * light.specular1.rgb;

        // only the last light applies fresnel
        if (slot == count - 1u && lookups[2].x != 0u) {
            if (light_flags.x != 0u) {
                diffuse_sum.a = values[2];
            }
            if (light_flags.y != 0u) {
                specular_sum.a = values[2];
            }
        }

        float facing = dot(light_vector, normal);
        facing = two_sided ? abs(facing) : max(facing, 0.0);
        float highlights = light_flags.z != 0u && facing == 0.0 ? 0.0 : 1.0;
        if ((flags & 12u) != 0u) {
            float length2 = dot(half_vector, half_vector);
            float factor = length2 == 0.0 ? 0.0 : min(facing / length2, 1.0);
            if ((flags & 4u) != 0u) {
                specular0 *= factor;
            }
            if ((flags & 8u) != 0u) {
                specular1 *= factor;
            }
        }

        bool shadowed = (flags & 64u) != 0u && (shadow_word & 1u) != 0u;
        vec3 shadow_primary = shadowed && (shadow_word & 0x200u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 shadow_secondary = shadowed && (shadow_word & 0x400u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 diffuse = (light.diffuse.rgb * facing + light.ambient.rgb) * distance * spot;
        vec3 specular = (specular0 + specular1) * highlights * distance * spot;
        diffuse_sum.rgb += diffuse * shadow_primary;
        specular_sum.rgb += specular * shadow_secondary;
    }

    if ((shadow_word & 0x800u) != 0u) {
        if (light_flags.x != 0u) {
            diffuse_sum.a *= shadow.a;
        }
        if (light_flags.y != 0u) {
            specular_sum.a *= shadow.a;
        }
    }
    diffuse_sum.rgb += global_ambient.rgb;
    diffuse_out = vec4(quantize(diffuse_sum.r), quantize(diffuse_sum.g), quantize(diffuse_sum.b), quantize(diffuse_sum.a));
    specular_out = vec4(quantize(specular_sum.r), quantize(specular_sum.g), quantize(specular_sum.b), quantize(specular_sum.a));
}

precise vec3 color_operand(vec4 s, uint operand) {
    switch (operand) {
        case 0x1u: return vec3(1.0) - s.rgb;
        case 0x2u: return vec3(s.a);
        case 0x3u: return vec3(1.0 - s.a);
        case 0x4u: return vec3(s.r);
        case 0x5u: return vec3(1.0 - s.r);
        case 0x8u: return vec3(s.g);
        case 0x9u: return vec3(1.0 - s.g);
        case 0xCu: return vec3(s.b);
        case 0xDu: return vec3(1.0 - s.b);
        default: return s.rgb;
    }
}

precise float alpha_operand(vec4 s, uint operand) {
    switch (operand) {
        case 0x0u: return s.a;
        case 0x1u: return 1.0 - s.a;
        case 0x2u: return s.r;
        case 0x3u: return 1.0 - s.r;
        case 0x4u: return s.g;
        case 0x5u: return 1.0 - s.g;
        case 0x6u: return s.b;
        default: return 1.0 - s.b;
    }
}

// the operations' results are precise, so the driver fuses no multiply
// into an add, which it would do one way where it knows the operation and
// another where the generic shader does not, rounding differently
vec3 combine_rgb(uint op, vec3 a, vec3 b, vec3 c) {
    precise vec3 result;
    switch (op) {
        case 0u: result = a; break;
        case 1u: result = a * b; break;
        case 2u: result = min(a + b, vec3(1.0)); break;
        case 3u: result = clamp(a + b - 0.5, 0.0, 1.0); break;
        case 4u: result = a * c + b * (vec3(1.0) - c); break;
        case 5u: result = max(a - b, vec3(0.0)); break;
        case 6u:
        case 7u: {
            // both inputs are signed values packed into 0..1
            float d = clamp(dot(a * 2.0 - 1.0, b * 2.0 - 1.0), 0.0, 1.0);
            result = vec3(d);
            break;
        }
        case 8u: result = min(a * b + c, vec3(1.0)); break;
        default: result = min(a + b, vec3(1.0)) * c; break;
    }
    return result;
}

float combine_alpha(uint op, float a, float b, float c) {
    precise float result;
    switch (op) {
        case 0u: result = a; break;
        case 1u: result = a * b; break;
        case 2u: result = min(a + b, 1.0); break;
        case 3u: result = clamp(a + b - 0.5, 0.0, 1.0); break;
        case 4u: result = a * c + b * (1.0 - c); break;
        case 5u: result = max(a - b, 0.0); break;
        case 6u:
        case 7u: result = a; break;
        case 8u: result = min(a * b + c, 1.0); break;
        default: result = min(a + b, 1.0) * c; break;
    }
    return result;
}

uint operation(uint raw) {
    uint op = raw & 0xFu;
    return op > 9u ? 9u : op;
}

float scale_factor(uint raw) {
    uint s = raw & 3u;
    return s == 1u ? 2.0 : s == 2u ? 4.0 : 1.0;
}

bool compare(uint function, float value, float reference) {
    switch (function) {
        case 0u: return false;
        case 1u: return true;
        case 2u: return value == reference;
        case 3u: return value != reference;
        case 4u: return value < reference;
        case 5u: return value <= reference;
        case 6u: return value > reference;
        default: return value >= reference;
    }
}

// a source's value for a combiner stage
vec4 source_value(uint s, vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 textures[4], vec4 previous, vec4 held, vec4 constant) {
    switch (s) {
        case 0x0u: return primary;
        case 0x1u: return fragment_primary;
        case 0x2u: return fragment_secondary;
        case 0x3u: return textures[0];
        case 0x4u: return textures[1];
        case 0x5u: return textures[2];
        case 0x6u: return textures[3];
        case 0xDu: return held;
        case 0xEu: return constant;
        default: return previous;
    }
}

// one combiner stage, its registers and the buffer update register
// constant in each pipeline but the generic one
void combine_stage(
    uint stage, uvec4 registers, uint update,
    vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 textures[4],
    inout vec4 previous, inout vec4 held, inout vec4 next_buffer
) {
    uint source = registers.x;
    uint operand = registers.y;
    uint combiner = registers.z;
    uint scale = registers.w;
    // a stage handing the previous color on as it is changes nothing, as
    // the color is between 0 and 1 already. titles leave most stages so,
    // and the generic shader, which cannot drop them as it compiles, skips
    // them as it runs
    bool passes = (source & 0x000F000Fu) == 0x000F000Fu && (operand & 0x700Fu) == 0u
        && (combiner & 0x000F000Fu) == 0u && scale == 0u;
    if (!passes) {
        vec4 constant = unpack_color(tev[stage * 2u].w);
        vec4 rgb_in[3];
        vec4 alpha_in[3];
        for (uint i = 0u; i < 3u; i++) {
            rgb_in[i] = source_value((source >> (i * 4u)) & 0xFu, primary, fragment_primary, fragment_secondary, textures, previous, held, constant);
            alpha_in[i] = source_value((source >> (16u + i * 4u)) & 0xFu, primary, fragment_primary, fragment_secondary, textures, previous, held, constant);
        }
        uint color_op = operation(combiner);
        uint alpha_op = operation(combiner >> 16);
        vec3 rgb = combine_rgb(
            color_op,
            color_operand(rgb_in[0], operand & 0xFu),
            color_operand(rgb_in[1], (operand >> 4) & 0xFu),
            color_operand(rgb_in[2], (operand >> 8) & 0xFu)
        );
        float alpha = color_op == 7u
            ? rgb.r
            : combine_alpha(
                alpha_op,
                alpha_operand(alpha_in[0], (operand >> 12) & 0x7u),
                alpha_operand(alpha_in[1], (operand >> 16) & 0x7u),
                alpha_operand(alpha_in[2], (operand >> 20) & 0x7u)
            );
        precise vec4 scaled = clamp(vec4(rgb * scale_factor(scale), alpha * scale_factor(scale >> 16)), 0.0, 1.0);
        previous = scaled;
    }

    held = next_buffer;
    if (stage < 4u) {
        if ((update & (0x100u << stage)) != 0u) {
            next_buffer.rgb = previous.rgb;
        }
        if ((update & (0x1000u << stage)) != 0u) {
            next_buffer.a = previous.a;
        }
    }
}

void main() {
    vec4 primary = clamp(in_color, 0.0, 1.0);

    // the units the configuration turns on, unit 2 can read coordinate
    // set 1 instead of its own
    uint texture_config = DYNAMIC != 0u ? (misc.w & 0x2307u) | (TEXTURE_CONFIG & 0x400u) : TEXTURE_CONFIG;
    vec4 textures[4] = vec4[4](vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0));
    if ((texture_config & 1u) != 0u) {
        textures[0] = sample_unit(0u, in_texcoords01.xy);
    }
    if ((texture_config & 2u) != 0u) {
        textures[1] = sample_unit(1u, in_texcoords01.zw);
    }
    if ((texture_config & 4u) != 0u) {
        textures[2] = sample_unit(2u, (texture_config & (1u << 13)) != 0u ? in_texcoords01.zw : in_texcoord2);
    }
    if ((texture_config & (1u << 10)) != 0u) {
        uint set = min((texture_config >> 8) & 3u, 2u);
        textures[3] = procedural(set == 0u ? in_texcoords01.xy : (set == 1u ? in_texcoords01.zw : in_texcoord2));
    }

    // without fragment lighting the primary fragment color is the vertex
    // color and there is no specular term
    vec4 fragment_primary = primary;
    vec4 fragment_secondary = vec4(0.0, 0.0, 0.0, 1.0);
    if (LIGHTING != 0u) {
        shade(textures, fragment_primary, fragment_secondary);
    }

    vec4 previous = primary;
    // the buffer lags a stage behind, the first stage reads zero, the second
    // the configured buffer color
    vec4 held = vec4(0.0);
    vec4 next_buffer = unpack_color(misc.y);
    uint update = DYNAMIC != 0u ? misc.x & 0xFF00u : UPDATE;
    combine_stage(0u, stage_registers(0u, uvec4(SOURCE0, OPERAND0, COMBINER0, SCALE0)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(1u, stage_registers(1u, uvec4(SOURCE1, OPERAND1, COMBINER1, SCALE1)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(2u, stage_registers(2u, uvec4(SOURCE2, OPERAND2, COMBINER2, SCALE2)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(3u, stage_registers(3u, uvec4(SOURCE3, OPERAND3, COMBINER3, SCALE3)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(4u, stage_registers(4u, uvec4(SOURCE4, OPERAND4, COMBINER4, SCALE4)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(5u, stage_registers(5u, uvec4(SOURCE5, OPERAND5, COMBINER5, SCALE5)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);

    // the color leaves as whole bytes, the way the software path truncates.
    // precise, as everything from the operands on is, or where the driver
    // knows the operands it works 1 - a times 255 out as 255 - 255a, which
    // rounds to the other side of a byte
    precise vec4 color = floor(previous * 255.0);
    uint alpha_test = DYNAMIC != 0u ? misc.z & 0x71u : ALPHA_TEST;
    if ((alpha_test & 1u) != 0u && !compare((alpha_test >> 4) & 7u, color.a, float((misc.z >> 8) & 0xFFu))) {
        discard;
    }
    precise vec4 written = color / 255.0;
    out_color = written;

    // what the GPU shaded maps z/w itself, which the clipper got exactly.
    // precise as the combiners are
    precise float depth = in_depth;
    uint depth_mode = DYNAMIC != 0u ? flags.x : DEPTH_MODE;
    if ((depth_mode & 2u) != 0u) {
        depth = -gl_FragCoord.z * uintBitsToFloat(flags.z) + uintBitsToFloat(flags.w);
    }
    if ((depth_mode & 1u) != 0u) {
        depth /= gl_FragCoord.w;
    }
    gl_FragDepth = clamp(depth, 0.0, 1.0);
}
