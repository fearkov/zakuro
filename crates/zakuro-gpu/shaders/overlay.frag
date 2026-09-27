#version 450
// colors come premultiplied and in sRGB, a target that expects linear
// values gets them converted.
layout(push_constant) uniform Screen {
    vec2 size;
    uint srgb;
} screen;

layout(set = 0, binding = 0) uniform sampler2D image;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 out_color;

vec3 linear_from_srgb(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
    vec4 color = frag_color * texture(image, frag_uv);
    if (screen.srgb != 0u) {
        color.rgb = linear_from_srgb(color.rgb);
    }
    out_color = color;
}
