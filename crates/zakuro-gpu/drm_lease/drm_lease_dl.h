// wayland-client and libdrm, opened at run time: the game then starts where they are missing, and they are only
// loaded with the feature on. The macros before wayland-client-protocol.h are SDL's way of doing this
// (src/video/wayland/SDL_waylanddyn.h, zlib license): the header's inline functions call through the pointers, and
// its "extern const struct wl_interface x;" lines turn into the pointers' own declarations.
#pragma once
#include <wayland-client-core.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define DRM_LEASE_DL_FUNCS(X)                                                                      \
    X(wl_display_connect) X(wl_display_disconnect) X(wl_display_get_fd) X(wl_display_flush)        \
    X(wl_display_prepare_read) X(wl_display_cancel_read) X(wl_display_read_events)                 \
    X(wl_display_dispatch_pending) X(wl_proxy_marshal_flags) X(wl_proxy_add_listener)              \
    X(wl_proxy_destroy) X(wl_proxy_get_version)
#define DRM_LEASE_DL_DRM_FUNCS(X)                                                                  \
    X(drmModeGetConnector) X(drmModeFreeConnector) X(drmModeObjectGetProperties)                   \
    X(drmModeFreeObjectProperties) X(drmModeGetProperty) X(drmModeFreeProperty)                   \
    X(drmModeGetConnectorCurrent) X(drmModeGetResources) X(drmModeFreeResources) X(drmSetClientCap) \
    X(drmModeCreatePropertyBlob) X(drmModeDestroyPropertyBlob) X(drmModeAtomicAlloc)               \
    X(drmModeAtomicAddProperty) X(drmModeAtomicCommit) X(drmModeAtomicFree)

#define X(name) extern decltype(&::name) dl_##name;
extern "C" {
DRM_LEASE_DL_FUNCS(X)
DRM_LEASE_DL_DRM_FUNCS(X)
extern const struct wl_interface* dl_wl_registry_interface;
extern const struct wl_interface* dl_wl_callback_interface;
}
#undef X

#define wl_display_connect (*dl_wl_display_connect)
#define wl_display_disconnect (*dl_wl_display_disconnect)
#define wl_display_get_fd (*dl_wl_display_get_fd)
#define wl_display_flush (*dl_wl_display_flush)
#define wl_display_prepare_read (*dl_wl_display_prepare_read)
#define wl_display_cancel_read (*dl_wl_display_cancel_read)
#define wl_display_read_events (*dl_wl_display_read_events)
#define wl_display_dispatch_pending (*dl_wl_display_dispatch_pending)
#define wl_proxy_marshal_flags (*dl_wl_proxy_marshal_flags)
#define wl_proxy_add_listener (*dl_wl_proxy_add_listener)
#define wl_proxy_destroy (*dl_wl_proxy_destroy)
#define wl_proxy_get_version (*dl_wl_proxy_get_version)
#define drmModeGetConnector (*dl_drmModeGetConnector)
#define drmModeFreeConnector (*dl_drmModeFreeConnector)
#define drmModeObjectGetProperties (*dl_drmModeObjectGetProperties)
#define drmModeFreeObjectProperties (*dl_drmModeFreeObjectProperties)
#define drmModeGetProperty (*dl_drmModeGetProperty)
#define drmModeFreeProperty (*dl_drmModeFreeProperty)
#define drmModeGetConnectorCurrent (*dl_drmModeGetConnectorCurrent)
#define drmModeGetResources (*dl_drmModeGetResources)
#define drmModeFreeResources (*dl_drmModeFreeResources)
#define drmSetClientCap (*dl_drmSetClientCap)
#define drmModeCreatePropertyBlob (*dl_drmModeCreatePropertyBlob)
#define drmModeDestroyPropertyBlob (*dl_drmModeDestroyPropertyBlob)
#define drmModeAtomicAlloc (*dl_drmModeAtomicAlloc)
#define drmModeAtomicAddProperty (*dl_drmModeAtomicAddProperty)
#define drmModeAtomicCommit (*dl_drmModeAtomicCommit)
#define drmModeAtomicFree (*dl_drmModeAtomicFree)
#define wl_registry_interface (*dl_wl_registry_interface)
#define wl_callback_interface (*dl_wl_callback_interface)
#include <wayland-client-protocol.h>

bool drm_lease_dl_open();  // false when a library or function is missing (logged)
