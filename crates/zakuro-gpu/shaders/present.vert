#version 450
// A single oversized triangle covers the viewport with no vertex buffer.
layout(location = 0) out vec2 uv;
layout(push_constant) uniform Attributes {
    layout(offset = 0) int orientation;
} attributes;
void main() {
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = position;
    if (attributes.orientation == 90) {
        uv = vec2(uv.y, 1.0 - uv.x);
    } else if (attributes.orientation == 180) {
        uv = vec2(1.0 - uv.x, 1.0 - uv.y);
    } else if (attributes.orientation == 270) {
        uv = vec2(1.0 - uv.y, uv.x);
    }
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
