# OpenGL rasterizer TODO

The current MVP only rasterizes untextured, vertex-colored triangles. Unsupported draws use Zakuro's complete CPU rasterizer.

Not implemented in the OpenGL path:

- PICA vertex shaders and geometry shaders on the GPU; vertex processing, clipping, and culling still run on the CPU.
- Texture sampling, mipmaps, texture replacement, and non-passthrough texture-environment (TEV) combiners.
- Fragment lighting, procedural textures, fog, and alpha testing.
- Blending, logic operations, and partial color masks; OpenGL depth/stencil currently supports D16, D24, and D24S8 only when the color path is otherwise supported.
- Framebuffer formats other than RGBA8 and dimensions that are not multiples of 8.
- Resolution scaling, direct GPU presentation, and frame interpolation.
- Full-range synchronization is used for an overlapping target; finer dirty tile/rectangle tracking could reduce writeback cost for sparse draws.
- OpenGL render-target cache eviction and GL object destruction are not yet bounded; long sessions with many distinct targets can retain resources.
- Direct GPU display transfer/presentation is not implemented, so transfers and scanout still synchronize to guest memory.
- Partial overlap and aliasing between differently sized color/depth views remain conservative: synchronization resolves full cached attachments, and unusual overlapping views should be covered by additional integration cases.
- OpenGL depth/stencil readback depends on driver support for the selected packed depth formats; unsupported/incomplete FBOs fall back to the CPU path.

# Next work

See [PLAN.md](PLAN.md) for the current implementation plan.
