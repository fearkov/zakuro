#version 450
// compiled twice, as it is and with STEREO defined, which draws the top
// screen's two pictures in one for 3D:
//   glslc present.frag -o present.frag.spv
//   glslc -DSTEREO present.frag -o present_stereo.frag.spv
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;
layout(set = 0, binding = 0) uniform sampler2D screen;
#ifdef STEREO
layout(set = 0, binding = 1) uniform sampler2D right_eye;
#endif
// where the screen lies in the image, its corner and size as texture
// coordinates, then the corners inset by half a texel, which samples keep
// within so filtering takes nothing from around it. the whole image when it
// holds just the screen. then how a screen drawn bigger than its pixels is
// filtered: 0 smooth, 1 pixels, 2 sharp.
layout(push_constant) uniform Crop {
    layout(offset = 16) vec4 area;
    vec4 bounds;
    int mode;
#ifdef STEREO
    // how the eyes are shown, 1 in red and cyan, 2 on alternate rows,
    // whether they swap, then where the right eye's picture lies
    int stereo;
    int swap;
    int unused;
    vec4 right_area;
    vec4 right_bounds;
#endif
} crop;

// a screen drawn bigger than the window shows it is averaged over every
// texel a window pixel covers, which smooths its edges the way drawing at a
// higher resolution should. one drawn smaller is filtered as asked. a
// little over a texel a pixel still counts as one, at 1:1 the derivatives
// land on either side of it.
vec3 shown(sampler2D image, vec4 area, vec4 bounds) {
    vec2 at = area.xy + uv * area.zw;
    vec2 size = vec2(textureSize(image, 0));
    vec2 covered = fwidth(at) * size;
    if (max(covered.x, covered.y) <= 1.001) {
        vec2 texel = at * size;
        if (crop.mode == 1) {
            // the nearest texel, its middle sampled
            at = (floor(texel) + 0.5) / size;
        } else if (crop.mode == 2) {
            // bilinear between texel centres, its blend squeezed into the
            // one window pixel each texel's edge falls in, at any scale
            vec2 scale = 1.0 / max(covered, vec2(1e-6));
            vec2 centred = texel - 0.5;
            vec2 blend = clamp((fract(centred) - 0.5) * scale + 0.5, 0.0, 1.0);
            at = (floor(centred) + 0.5 + blend) / size;
        }
        return texture(image, clamp(at, bounds.xy, bounds.zw)).rgb;
    }
    // each sample is bilinear, so it already averages its four texels
    ivec2 taps = ivec2(clamp(ceil(covered), 1.0, 4.0));
    vec3 sum = vec3(0.0);
    for (int y = 0; y < taps.y; y++) {
        for (int x = 0; x < taps.x; x++) {
            vec2 offset = ((vec2(x, y) + 0.5) / vec2(taps) - 0.5) * covered / size;
            sum += texture(image, clamp(at + offset, bounds.xy, bounds.zw)).rgb;
        }
    }
    return sum / float(taps.x * taps.y);
}

void main() {
#ifdef STEREO
    vec3 left = shown(screen, crop.area, crop.bounds);
    vec3 right = shown(right_eye, crop.right_area, crop.right_bounds);
    if (crop.swap != 0) {
        vec3 other = left;
        left = right;
        right = other;
    }
    if (crop.stereo == 1) {
        // Dubois' red and cyan projection, least squares over what the
        // glasses let through, with less ghosting than the plain split
        vec3 red = vec3(dot(left, vec3(0.437, 0.449, 0.164)), dot(left, vec3(-0.062, -0.062, -0.024)), dot(left, vec3(-0.048, -0.050, -0.017)));
        vec3 cyan = vec3(dot(right, vec3(-0.011, -0.032, -0.007)), dot(right, vec3(0.377, 0.761, 0.009)), dot(right, vec3(-0.026, -0.093, 1.234)));
        colour = vec4(clamp(red + cyan, 0.0, 1.0), 1.0);
    } else {
        // window rows from the top, even ones the left eye's
        colour = vec4((int(gl_FragCoord.y) & 1) == 0 ? left : right, 1.0);
    }
#else
    colour = vec4(shown(screen, crop.area, crop.bounds), 1.0);
#endif
}
