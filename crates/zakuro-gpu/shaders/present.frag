#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;
layout(set = 0, binding = 0) uniform sampler2D screen;
void main() {
    colour = vec4(texture(screen, uv).rgb, 1.0);
}
