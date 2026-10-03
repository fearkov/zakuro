#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;
layout(set = 0, binding = 0) uniform sampler2D screen;
// where the screen lies in the image, its corner and size as texture
// coordinates, then the corners inset by half a texel, which samples keep
// within so filtering takes nothing from around it. the whole image when it
// holds just the screen.
layout(push_constant) uniform Crop {
    vec4 area;
    vec4 bounds;
} crop;
// a screen drawn bigger than the window shows it is averaged over every
// texel a window pixel covers, which smooths its edges the way drawing at a
// higher resolution should. one drawn smaller is filtered as usual.
void main() {
    vec2 at = crop.area.xy + uv * crop.area.zw;
    vec2 size = vec2(textureSize(screen, 0));
    vec2 covered = fwidth(at) * size;
    if (max(covered.x, covered.y) <= 1.0) {
        colour = vec4(texture(screen, clamp(at, crop.bounds.xy, crop.bounds.zw)).rgb, 1.0);
        return;
    }
    // each sample is bilinear, so it already averages its four texels
    ivec2 taps = ivec2(clamp(ceil(covered), 1.0, 4.0));
    vec3 sum = vec3(0.0);
    for (int y = 0; y < taps.y; y++) {
        for (int x = 0; x < taps.x; x++) {
            vec2 offset = ((vec2(x, y) + 0.5) / vec2(taps) - 0.5) * covered / size;
            sum += texture(screen, clamp(at + offset, crop.bounds.xy, crop.bounds.zw)).rgb;
        }
    }
    colour = vec4(sum / float(taps.x * taps.y), 1.0);
}
