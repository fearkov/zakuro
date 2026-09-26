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
    // x, a w-buffer, y, lighting is on
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
};

layout(std430, set = 0, binding = 4) readonly buffer Tables {
    // 24 tables of 256 entries, a value and the step to the next
    vec2 tables[];
};

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
    vec2 st = vec2(uv.x, 1.0 - uv.y);
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

float lookup(uint table, uint entry, float delta) {
    vec2 value = tables[table * 256u + entry];
    return value.x + value.y * delta;
}

vec3 rotate(vec4 q, vec3 v) {
    vec3 inner = cross(q.xyz, v);
    return v + 2.0 * cross(q.xyz, inner + v * q.w);
}

float quantize(float c) {
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

    vec4 diffuse_sum = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 specular_sum = vec4(0.0, 0.0, 0.0, 1.0);
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

vec3 color_operand(vec4 s, uint operand) {
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

float alpha_operand(vec4 s, uint operand) {
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

vec3 combine_rgb(uint op, vec3 a, vec3 b, vec3 c) {
    switch (op) {
        case 0u: return a;
        case 1u: return a * b;
        case 2u: return min(a + b, vec3(1.0));
        case 3u: return clamp(a + b - 0.5, 0.0, 1.0);
        case 4u: return a * c + b * (vec3(1.0) - c);
        case 5u: return max(a - b, vec3(0.0));
        case 6u:
        case 7u: {
            // both inputs are signed values packed into 0..1
            float d = clamp(dot(a * 2.0 - 1.0, b * 2.0 - 1.0), 0.0, 1.0);
            return vec3(d);
        }
        case 8u: return min(a * b + c, vec3(1.0));
        default: return min(a + b, vec3(1.0)) * c;
    }
}

float combine_alpha(uint op, float a, float b, float c) {
    switch (op) {
        case 0u: return a;
        case 1u: return a * b;
        case 2u: return min(a + b, 1.0);
        case 3u: return clamp(a + b - 0.5, 0.0, 1.0);
        case 4u: return a * c + b * (1.0 - c);
        case 5u: return max(a - b, 0.0);
        case 6u:
        case 7u: return a;
        case 8u: return min(a * b + c, 1.0);
        default: return min(a + b, 1.0) * c;
    }
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

void main() {
    vec4 primary = clamp(in_color, 0.0, 1.0);

    // the units the configuration turns on, unit 2 can read coordinate
    // set 1 instead of its own
    uint texture_config = misc.w;
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

    // without fragment lighting the primary fragment color is the vertex
    // color and there is no specular term
    vec4 fragment_primary = primary;
    vec4 fragment_secondary = vec4(0.0, 0.0, 0.0, 1.0);
    if (flags.y != 0u) {
        shade(textures, fragment_primary, fragment_secondary);
    }

    vec4 previous = primary;
    // the buffer lags a stage behind, the first stage reads zero, the second
    // the configured buffer color
    vec4 held = vec4(0.0);
    vec4 next_buffer = unpack_color(misc.y);
    uint update = misc.x;
    for (uint stage = 0u; stage < 6u; stage++) {
        uvec4 words = tev[stage * 2u];
        uint source = words.x;
        uint operand = words.y;
        uint combiner = words.z;
        vec4 constant = unpack_color(words.w);
        uint scale = tev[stage * 2u + 1u].x;

        vec4 inputs_rgb[3];
        vec4 inputs_alpha[3];
        for (uint i = 0u; i < 3u; i++) {
            uint sources[2] = uint[2]((source >> (i * 4u)) & 0xFu, (source >> (16u + i * 4u)) & 0xFu);
            for (uint which = 0u; which < 2u; which++) {
                uint s = sources[which];
                vec4 value;
                switch (s) {
                    case 0x0u: value = primary; break;
                    case 0x1u: value = fragment_primary; break;
                    case 0x2u: value = fragment_secondary; break;
                    case 0x3u: value = textures[0]; break;
                    case 0x4u: value = textures[1]; break;
                    case 0x5u: value = textures[2]; break;
                    case 0x6u: value = textures[3]; break;
                    case 0xDu: value = held; break;
                    case 0xEu: value = constant; break;
                    default: value = previous; break;
                }
                if (which == 0u) {
                    inputs_rgb[i] = value;
                } else {
                    inputs_alpha[i] = value;
                }
            }
        }

        uint color_op = operation(combiner);
        uint alpha_op = operation(combiner >> 16);
        vec3 rgb = combine_rgb(
            color_op,
            color_operand(inputs_rgb[0], operand & 0xFu),
            color_operand(inputs_rgb[1], (operand >> 4) & 0xFu),
            color_operand(inputs_rgb[2], (operand >> 8) & 0xFu)
        );
        float alpha = color_op == 7u
            ? rgb.r
            : combine_alpha(
                alpha_op,
                alpha_operand(inputs_alpha[0], (operand >> 12) & 0x7u),
                alpha_operand(inputs_alpha[1], (operand >> 16) & 0x7u),
                alpha_operand(inputs_alpha[2], (operand >> 20) & 0x7u)
            );
        previous = clamp(vec4(rgb * scale_factor(scale), alpha * scale_factor(scale >> 16)), 0.0, 1.0);

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

    // the color leaves as whole bytes, the way the software path truncates
    vec4 color = floor(previous * 255.0);
    uint alpha_test = misc.z;
    if ((alpha_test & 1u) != 0u && !compare((alpha_test >> 4) & 7u, color.a, float((alpha_test >> 8) & 0xFFu))) {
        discard;
    }
    out_color = color / 255.0;

    float depth = in_depth;
    if (flags.x != 0u) {
        depth /= gl_FragCoord.w;
    }
    gl_FragDepth = clamp(depth, 0.0, 1.0);
}
