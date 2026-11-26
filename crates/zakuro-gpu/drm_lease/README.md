The second panel's lease (drm-lease-v1) and its touchscreen, for src/backend/drm_lease.rs. Built by build.rs on Linux
when wayland-scanner and the wayland-client and libdrm headers are found. wayland-client and libdrm are opened at run
time, so the program does not link them.

These files are Azahar's DRM lease code (GPLv2 or later, license.txt) as MHGU Recomp's AArch64 Linux builds carry it,
taken from https://github.com/cerroverb/MHGURecomp-aarch64 (branch aarch64-linux-v2, commit 8c10fd6,
third_party/azahar-drm-lease). Each file's header names its Azahar source
(https://github.com/virtudude/azahar, branch drm-leasing, commit d16299140, and https://github.com/cerroverb/azahar,
branch drm-leasing-link-crtc, commit 71a132b3b) and the edits made for MHGU Recomp.

Edits for Zakuro
- the environment variables are named SECOND_SCREEN_ instead of MHGU_SECOND_DISPLAY_
- drm_display_surface.cpp is left out and drm_lease.h no longer declares its two functions. src/backend/vulkan_panel.rs
  makes the same Vulkan calls through ash
- comments name the Zakuro files that use these instead of MHGU Recomp's
