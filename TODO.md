# OpenGL rasterizer TODO

The current MVP only rasterizes untextured, vertex-colored triangles. Unsupported draws use Zakuro's complete CPU rasterizer.

Not implemented in the OpenGL path:

- PICA vertex shaders and geometry shaders on the GPU; vertex processing, clipping, and culling still run on the CPU.
- Texture sampling, mipmaps, texture replacement, and non-passthrough texture-environment (TEV) combiners.
- Fragment lighting, procedural textures, fog, and alpha testing.
- Depth and stencil tests or writes, blending, logic operations, and partial color masks.
- Framebuffer formats other than RGBA8 and dimensions that are not multiples of 8.
- Resolution scaling, cached render targets, direct GPU presentation, and frame interpolation.
- Fast GPU/CPU synchronization. The MVP uploads and reads back the full color buffer for every accelerated draw, so it may be slower than the CPU rasterizer even for supported draws.

# Next work

See [PLAN.md](PLAN.md) for the current implementation plan.
