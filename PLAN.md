# OpenGL rasterizer plan

## Current state

The MVP requires an OpenGL 4.3 context and accelerates draws that use RGBA8 targets with tile-aligned dimensions, write all color channels, and have no textures, lighting, fog, alpha test, blending, or logic operation. It supports depth/stencil tests and writes for D16, D24, and D24S8 attachments. The texture environment must be a passthrough. Zakuro still runs the vertex shader, clipping, and culling on the CPU; OpenGL rasterizes the resulting triangles and interpolates vertex color. Other draws use the existing CPU rasterizer.

The OpenGL path keeps keyed RGBA8/depth targets resident across draws. It synchronizes overlapping guest-memory ranges lazily through the existing GPU sync hooks. Depth/stencil formats D16, D24, and D24S8 are supported when the rest of the draw state is in the accelerated subset.

## Implementation sequence

1. Add focused comparison tests for OpenGL coverage, interpolation, scissoring, and tiled-buffer conversion against the CPU path. Run them with a software OpenGL context where available.
2. Add common texture formats and sampling. Reuse the existing texture decoder, then add mip level selection and PICA wrap/filter behavior.
3. Implement the TEV stages, alpha test, blending, and color masks. Keep CPU fallback for any unsupported state until each case is verified.
4. Move PICA vertex shader execution to OpenGL and handle geometry-shader draws through CPU preparation.
5. Add lighting, fog, procedural textures, and remaining framebuffer formats.
6. Keep keyed OpenGL color/depth targets resident across draws. Track dirty GPU attachments and invalidate them on CPU writes; resolve only on overlapping fallback draws, CPU reads/writes, fills, and transfers.
7. Add resolution scaling, pipeline/resource caching, texture-pack support on the GPU path, and assess direct presentation and frame interpolation separately.
8. Validate the complete path on OpenGL 4.3 hardware across supported operating systems; retain software fallback for unsupported drivers and unimplemented states.
