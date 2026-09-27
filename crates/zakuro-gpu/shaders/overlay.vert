#version 450
// the overlay's triangles, given in window pixels from the top left.
layout(push_constant) uniform Screen {
    vec2 size;
    uint srgb;
} screen;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

void main() {
    frag_uv = uv;
    frag_color = color;
    gl_Position = vec4(position / screen.size * 2.0 - 1.0, 0.0, 1.0);
}
