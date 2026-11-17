#version 450

// vertices the CPU already shaded, clipped and placed on the target, w/ w kept so the varyings interpolate w/ perspective
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec4 texcoords01;
layout(location = 3) in vec4 texcoord2_depth;
layout(location = 4) in vec4 quaternion;
layout(location = 5) in vec4 view;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_texcoords01;
layout(location = 2) out vec2 out_texcoord2;
// the mapped depth b4 any w buffer scaling, linear on the screen
layout(location = 3) noperspective out float out_depth;
layout(location = 4) out vec4 out_quaternion;
// the view vector, and in w texture coord set 0's w
layout(location = 5) out vec4 out_view;

void main() {
    // the mapped depth on the rasterizer's side too, for draws whose
    // fragment shader leaves depth alone
    gl_Position = vec4(position.xy, texcoord2_depth.z * position.w, position.w);
    out_color = color;
    out_texcoords01 = texcoords01;
    out_texcoord2 = texcoord2_depth.xy;
    out_depth = texcoord2_depth.z;
    out_quaternion = quaternion;
    out_view = view;
}
