//! register offsets, split between the external block the LCD and transfer
//! engines live in and the internal PICA register file command lists write.

// -- external registers, as byte offsets from 0x1EF00000 --------------------

pub const LCD_TOP_BASE: usize = 0x0400;
pub const LCD_BOTTOM_BASE: usize = 0x0500;

/// offsets within one LCD controller's block.
pub const LCD_FB_A_LEFT: usize = 0x68;
pub const LCD_FB_A_RIGHT: usize = 0x6C;
pub const LCD_FB_FORMAT: usize = 0x70;
pub const LCD_FB_SELECT: usize = 0x78;
pub const LCD_FB_STRIDE: usize = 0x90;
pub const LCD_FB_B_LEFT: usize = 0x94;
pub const LCD_FB_B_RIGHT: usize = 0x98;

// -- internal (PICA) registers, as word indices -----------------------------

pub const REG_FACE_CULLING: usize = 0x0040;
pub const REG_VIEWPORT_WIDTH: usize = 0x0041;
pub const REG_VIEWPORT_WIDTH_RECIPROCAL: usize = 0x0042;
pub const REG_VIEWPORT_HEIGHT: usize = 0x0043;
pub const REG_VIEWPORT_HEIGHT_RECIPROCAL: usize = 0x0044;
pub const REG_VIEWPORT_DEPTH_RANGE: usize = 0x004D;
pub const REG_VIEWPORT_DEPTH_NEAR: usize = 0x004E;
/// GPUREG_DEPTHMAP_ENABLE, 1 for a z-buffer, 0 for a w-buffer.
pub const REG_DEPTHMAP_ENABLE: usize = 0x006D;

/// number of vertex shader output registers in use, minus one.
pub const REG_SHADER_OUTPUT_TOTAL: usize = 0x004F;
/// seven registers mapping each output register's components to semantics.
pub const REG_SHADER_OUTPUT_MAP: usize = 0x0050;
pub const REG_SHADER_OUTPUT_MAP_END: usize = 0x0056;

pub const REG_VIEWPORT_XY: usize = 0x0068;

pub const REG_TEXTURE_CONFIG: usize = 0x0080;
pub const REG_TEXTURE0_BORDER_COLOR: usize = 0x0081;
pub const REG_TEXTURE0_DIMENSIONS: usize = 0x0082;
pub const REG_TEXTURE0_PARAMETERS: usize = 0x0083;
pub const REG_TEXTURE0_ADDRESS: usize = 0x0085;
pub const REG_TEXTURE0_FORMAT: usize = 0x008E;

/// first of the six texture-environment stages.
pub const REG_TEV_STAGE0: usize = 0x00C0;
pub const REG_TEV_STAGE_STRIDE: usize = 8;

pub const REG_BLEND_FUNC: usize = 0x0101;
/// selects blending or logic-op output, bit 8 turns the blender on.
pub const REG_COLOR_OPERATION: usize = 0x0100;
pub const REG_ALPHA_TEST: usize = 0x0104;
pub const REG_STENCIL_TEST: usize = 0x0105;
pub const REG_STENCIL_OP: usize = 0x0106;
pub const REG_DEPTH_COLOR_MASK: usize = 0x0107;
/// GPUREG_COLORBUFFER_WRITE, zero forbids color writes altogether.
pub const REG_COLOR_BUFFER_WRITE: usize = 0x0113;
/// GPUREG_DEPTHBUFFER_WRITE, zero forbids depth and stencil writes.
pub const REG_DEPTH_STENCIL_WRITE: usize = 0x0115;
pub const REG_COLOR_BUFFER_FORMAT: usize = 0x0117;
pub const REG_DEPTH_BUFFER_FORMAT: usize = 0x0116;
pub const REG_DEPTH_BUFFER_ADDRESS: usize = 0x011C;
pub const REG_COLOR_BUFFER_ADDRESS: usize = 0x011D;
pub const REG_FRAMEBUFFER_DIMENSIONS: usize = 0x011E;

// -- vertex attribute fetching ---------------------------------------------

/// physical base address of the attribute arrays, in units of eight bytes.
pub const REG_ATTRIBUTE_BASE: usize = 0x0200;
pub const REG_ATTRIBUTE_FORMAT_LOW: usize = 0x0201;
pub const REG_ATTRIBUTE_FORMAT_HIGH: usize = 0x0202;
/// twelve loaders of three registers each.
pub const REG_ATTRIBUTE_LOADER: usize = 0x0203;
pub const REG_ATTRIBUTE_LOADER_COUNT: usize = 12;
pub const REG_ATTRIBUTE_LOADER_STRIDE: usize = 3;

pub const REG_INDEX_ARRAY: usize = 0x0227;
/// GPUREG_CMDBUF_SIZE0/1 and ADDR0/1 describe two command buffers, in units of
/// eight bytes, writing GPUREG_CMDBUF_JUMP0/1 moves execution to one of them.
pub const REG_CMDBUF_SIZE0: usize = 0x0238;
pub const REG_CMDBUF_ADDR0: usize = 0x023A;
pub const REG_CMDBUF_JUMP0: usize = 0x023C;
pub const REG_CMDBUF_JUMP1: usize = 0x023D;
/// selects which attribute the fixed-value writes below set, 15 means
/// immediate-mode vertex submission instead.
pub const REG_FIXED_ATTRIBUTE_INDEX: usize = 0x0232;
pub const REG_FIXED_ATTRIBUTE_DATA: usize = 0x0233;
pub const REG_FIXED_ATTRIBUTE_DATA_END: usize = 0x0235;
pub const REG_VERTEX_COUNT: usize = 0x0228;
pub const REG_VERTEX_OFFSET: usize = 0x022A;

/// triggers a non-indexed draw.
pub const REG_DRAW_ARRAYS: usize = 0x022E;
/// triggers an indexed draw.
pub const REG_DRAW_ELEMENTS: usize = 0x022F;

/// GPUREG_GEOSTAGE_CONFIG, bits 0-1 are 2 when a geometry shader runs.
pub const REG_GEOSTAGE_CONFIG: usize = 0x0229;
/// GPUREG_VSH_NUM_ATTR, how many attributes an immediate-mode vertex has,
/// minus one.
pub const REG_VS_ATTRIBUTE_COUNT: usize = 0x0242;
/// GPUREG_VSH_COM_MODE, bit 0 set means the geometry shader unit is
/// configured on its own rather than mirroring the vertex shader's program.
pub const REG_VS_COM_MODE: usize = 0x0244;
/// GPUREG_VSH_OUTMAP_TOTAL1, how many attributes each vertex hands the
/// geometry shader, minus one.
pub const REG_VS_OUTPUT_TOTAL: usize = 0x024A;
/// GPUREG_GSH_MISC0, bits 0-7 select how vertices are grouped into
/// geometry shader invocations.
pub const REG_GS_CONFIG: usize = 0x0252;
pub const REG_PRIMITIVE_CONFIG: usize = 0x025E;

// -- shader unit blocks ------------------------------------------------------

pub const REG_GS_BLOCK: usize = 0x0280;
pub const REG_VS_BLOCK: usize = 0x02B0;
pub const SHADER_BLOCK_SIZE: usize = 0x2E;

pub const SHADER_BOOL_UNIFORMS: usize = 0x00;
pub const SHADER_INT_UNIFORMS: usize = 0x01;
pub const SHADER_INT_UNIFORMS_END: usize = 0x04;
pub const SHADER_INPUT_CONFIG: usize = 0x09;
pub const SHADER_ENTRY_POINT: usize = 0x0A;
pub const SHADER_INPUT_MAP_LOW: usize = 0x0B;
pub const SHADER_INPUT_MAP_HIGH: usize = 0x0C;
pub const SHADER_OUTPUT_MASK: usize = 0x0D;
pub const SHADER_UNIFORM_INDEX: usize = 0x10;
pub const SHADER_UNIFORM_DATA: usize = 0x11;
pub const SHADER_UNIFORM_DATA_END: usize = 0x18;
pub const SHADER_PROGRAM_INDEX: usize = 0x1B;
pub const SHADER_PROGRAM_DATA: usize = 0x1C;
pub const SHADER_PROGRAM_DATA_END: usize = 0x23;
pub const SHADER_DESCRIPTOR_INDEX: usize = 0x25;
pub const SHADER_DESCRIPTOR_DATA: usize = 0x26;
pub const SHADER_DESCRIPTOR_DATA_END: usize = 0x2D;

// -- vertex shader block ----------------------------------------------------

pub const REG_VS_NUM_INPUT_ATTRIBUTES: usize = REG_VS_BLOCK + SHADER_INPUT_CONFIG;
pub const REG_VS_INPUT_REGISTER_MAP_LOW: usize = REG_VS_BLOCK + SHADER_INPUT_MAP_LOW;
pub const REG_VS_OUTPUT_MASK: usize = REG_VS_BLOCK + SHADER_OUTPUT_MASK;
