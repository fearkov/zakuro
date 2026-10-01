#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;
layout(set = 0, binding = 0) uniform sampler2D screen;
// a screen drawn bigger than the window shows it is averaged over every
// texel a window pixel covers, which smooths its edges the way drawing at a
// higher resolution should. one drawn smaller is filtered as usual.
void main() {
    vec2 size = vec2(textureSize(screen, 0));
    vec2 covered = fwidth(uv) * size;
    if (max(covered.x, covered.y) <= 1.0) {
        colour = vec4(texture(screen, uv).rgb, 1.0);
        return;
    }
    // each sample is bilinear, so it already averages its four texels
    ivec2 taps = ivec2(clamp(ceil(covered), 1.0, 4.0));
    vec3 sum = vec3(0.0);
    for (int y = 0; y < taps.y; y++) {
        for (int x = 0; x < taps.x; x++) {
            vec2 offset = ((vec2(x, y) + 0.5) / vec2(taps) - 0.5) * covered / size;
            sum += texture(screen, uv + offset).rgb;
        }
    }
    colour = vec4(sum / float(taps.x * taps.y), 1.0);
}
