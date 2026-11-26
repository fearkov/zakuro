// See drm_lease_dl.h. New glue for MHGU Recomp (MIT).
#include <wayland-client-core.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <dlfcn.h>
#include <cstdio>

#include "drm_lease_dl.h"

#undef wl_registry_interface
#undef wl_callback_interface
#define X(name) decltype(&::name) dl_##name;
extern "C" {
#undef wl_display_connect
#undef wl_display_disconnect
#undef wl_display_get_fd
#undef wl_display_flush
#undef wl_display_prepare_read
#undef wl_display_cancel_read
#undef wl_display_read_events
#undef wl_display_dispatch_pending
#undef wl_proxy_marshal_flags
#undef wl_proxy_add_listener
#undef wl_proxy_destroy
#undef wl_proxy_get_version
#undef drmModeGetConnector
#undef drmModeFreeConnector
#undef drmModeObjectGetProperties
#undef drmModeFreeObjectProperties
#undef drmModeGetProperty
#undef drmModeFreeProperty
#undef drmModeGetConnectorCurrent
#undef drmModeGetResources
#undef drmModeFreeResources
#undef drmSetClientCap
#undef drmModeCreatePropertyBlob
#undef drmModeDestroyPropertyBlob
#undef drmModeAtomicAlloc
#undef drmModeAtomicAddProperty
#undef drmModeAtomicCommit
#undef drmModeAtomicFree
DRM_LEASE_DL_FUNCS(X)
DRM_LEASE_DL_DRM_FUNCS(X)
const struct wl_interface* dl_wl_registry_interface;
const struct wl_interface* dl_wl_callback_interface;
}
#undef X

static bool resolve(void* lib, const char* name, void** out) {
    *out = dlsym(lib, name);
    if (!*out) fprintf(stderr, "[second-display] %s is missing\n", name);
    return *out != nullptr;
}

bool drm_lease_dl_open() {
    void* wl = dlopen("libwayland-client.so.0", RTLD_NOW | RTLD_LOCAL);
    void* drm = dlopen("libdrm.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!wl || !drm) {
        fprintf(stderr, "[second-display] can't open wayland-client or libdrm: %s\n", dlerror());
        return false;
    }
    bool ok = true;
#define X(name) ok = resolve(wl, #name, reinterpret_cast<void**>(&dl_##name)) && ok;
    DRM_LEASE_DL_FUNCS(X)
#undef X
#define X(name) ok = resolve(drm, #name, reinterpret_cast<void**>(&dl_##name)) && ok;
    DRM_LEASE_DL_DRM_FUNCS(X)
#undef X
    ok = resolve(wl, "wl_registry_interface", const_cast<void**>(reinterpret_cast<const void**>(&dl_wl_registry_interface))) && ok;
    ok = resolve(wl, "wl_callback_interface", const_cast<void**>(reinterpret_cast<const void**>(&dl_wl_callback_interface))) && ok;
    return ok;
}
