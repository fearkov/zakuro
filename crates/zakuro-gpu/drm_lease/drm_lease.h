// The lease of the second panel and its touchscreen, for src/backend/drm_lease.rs. New glue for MHGU Recomp (MIT); the
// code behind it is Azahar's drm_lease_window.cpp. Zakuro makes the Vulkan surface in Rust (src/backend/vulkan_panel.rs),
// so drm_display_surface.cpp and its two functions are left out here.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;                 // the lease: stays open until drm_lease_release
    uint32_t connector_id;
    uint32_t width, height; // the panel's mode, as the panel scans it out
    int rotation;           // degrees: SECOND_SCREEN_ROTATION, else the connector's panel orientation
} DrmLeaseOutput;

// Asks the compositor for a connector (NULL or "": the first internal panel it offers). Blocks up to two timeouts of
// 5 s, so Zakuro runs it on a thread of its own. touch_device: NULL or "" reads no touch; else an evdev path, a device
// name or "auto".
bool drm_lease_acquire(const char *connector, int rotation, const char *touch_device, DrmLeaseOutput *out);
// Would drm_lease_acquire find that connector? Asks the compositor without leasing (blocks as drm_lease_acquire can).
bool drm_lease_offered(const char *connector);
bool drm_lease_alive(void);  // reads the compositor's events without waiting; false once the lease has ended
void drm_lease_release(void);

// The touch reader's thread: a finger at panel pixel x, y, or lifted. Defined by the caller (src/backend/drm_lease.rs).
void second_display_touch(bool down, unsigned x, unsigned y);

#ifdef __cplusplus
}
#endif
