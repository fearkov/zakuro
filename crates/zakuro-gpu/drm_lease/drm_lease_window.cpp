// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
// Ported from Azahar-Enhanced (https://github.com/virtudude/azahar, branch drm-leasing) commit d16299140,
// src/citra_qt/drm_lease/drm_lease_window.cpp. ReadConnectorMode, FindPropertyId and LinkConnectorToLeasedCrtc from cerroverb/azahar
// (https://github.com/cerroverb/azahar, branch drm-leasing-link-crtc) commit 71a132b3b, unchanged. Edits for MHGU Recomp:
// ReadPanelOrientation unchanged; the EmuWindow class gives way to the plain C functions at the end, whose steps follow DrmLeaseWindow::Initialize.

#include <cstring>
#include <memory>
#include <string>

#include "drm_lease_dl.h"
#include "drm_lease.h"
#include "drm_lease_client.h"
#include "drm_lease_touch.h"

namespace Frontend {

namespace {

// The preferred (or first) mode of the leased connector; VK_KHR_display picks
// its mode independently, this sizes the framebuffer layout and links the CRTC.
bool ReadConnectorMode(int lease_fd, u32 connector_id, drmModeModeInfo& out_mode) {
    drmModeConnector* connector = drmModeGetConnector(lease_fd, connector_id);
    if (!connector || connector->count_modes < 1) {
        if (connector) {
            drmModeFreeConnector(connector);
        }
        return false;
    }

    drmModeModeInfo mode = connector->modes[0];
    for (int i = 0; i < connector->count_modes; i++) {
        if (connector->modes[i].type & DRM_MODE_TYPE_PREFERRED) {
            mode = connector->modes[i];
            break;
        }
    }

    out_mode = mode;
    drmModeFreeConnector(connector);
    return true;
}

u32 FindPropertyId(int lease_fd, u32 object_id, u32 object_type, const char* name) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(lease_fd, object_id, object_type);
    if (!props) {
        return 0;
    }

    u32 property_id = 0;
    for (u32 i = 0; i < props->count_props && property_id == 0; i++) {
        drmModePropertyRes* prop = drmModeGetProperty(lease_fd, props->props[i]);
        if (!prop) {
            continue;
        }
        if (std::strcmp(prop->name, name) == 0) {
            property_id = prop->prop_id;
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return property_id;
}

// An idle compositor leaves the connector unlinked, and Mesa then accepts a
// CRTC only if drmModeGetCrtc reports no framebuffer. That call reads the
// CRTC's legacy primary plane, which need not be the leased plane and may be
// scanning out for the compositor, so vkGetPhysicalDeviceSurfaceFormatsKHR can
// return no formats. Link the connector first, with the CRTC left inactive so
// that no framebuffer is needed and the panel stays off until the first present.
// The property sequence follows drm_atomic_commit in Mesa's wsi_common_display.c.
void LinkConnectorToLeasedCrtc(int lease_fd, u32 connector_id, drmModeModeInfo mode) {
    drmModeConnector* connector = drmModeGetConnectorCurrent(lease_fd, connector_id);
    if (!connector) {
        return;
    }
    const bool linked = connector->encoder_id != 0;
    drmModeFreeConnector(connector);
    if (linked) {
        return;
    }

    drmModeRes* resources = drmModeGetResources(lease_fd);
    if (!resources) {
        return;
    }
    const u32 crtc_id = resources->count_crtcs > 0 ? resources->crtcs[0] : 0;
    drmModeFreeResources(resources);
    if (crtc_id == 0) {
        LOG_WARNING(Frontend, "drm-lease: the lease holds no CRTC");
        return;
    }

    if (drmSetClientCap(lease_fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
        LOG_WARNING(Frontend, "drm-lease: atomic modesetting is unavailable");
        return;
    }

    const u32 connector_crtc =
        FindPropertyId(lease_fd, connector_id, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID");
    const u32 crtc_mode = FindPropertyId(lease_fd, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID");
    const u32 crtc_active = FindPropertyId(lease_fd, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE");
    if (connector_crtc == 0 || crtc_mode == 0 || crtc_active == 0) {
        LOG_WARNING(Frontend, "drm-lease: missing atomic properties for the leased objects");
        return;
    }

    u32 mode_blob = 0;
    if (drmModeCreatePropertyBlob(lease_fd, &mode, sizeof(mode), &mode_blob) != 0) {
        LOG_WARNING(Frontend, "drm-lease: failed to create the mode blob");
        return;
    }

    drmModeAtomicReq* request = drmModeAtomicAlloc();
    if (request) {
        drmModeAtomicAddProperty(request, connector_id, connector_crtc, crtc_id);
        drmModeAtomicAddProperty(request, crtc_id, crtc_mode, mode_blob);
        drmModeAtomicAddProperty(request, crtc_id, crtc_active, 0);
        const int ret =
            drmModeAtomicCommit(lease_fd, request, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
        if (ret != 0) {
            LOG_WARNING(Frontend, "drm-lease: failed to link connector {} to CRTC {}: {}",
                        connector_id, crtc_id, std::strerror(-ret));
        } else {
            LOG_INFO(Frontend, "drm-lease: linked connector {} to CRTC {}", connector_id, crtc_id);
        }
        drmModeAtomicFree(request);
    }
    drmModeDestroyPropertyBlob(lease_fd, mode_blob);
}

// Content rotation from the connector's "panel orientation" property, or -1
// when absent. Left/Right Side Up map per the compositor convention.
int ReadPanelOrientation(int lease_fd, u32 connector_id) {
    drmModeObjectProperties* props =
        drmModeObjectGetProperties(lease_fd, connector_id, DRM_MODE_OBJECT_CONNECTOR);
    if (!props) {
        return -1;
    }

    int rotation = -1;
    for (u32 i = 0; i < props->count_props && rotation < 0; i++) {
        drmModePropertyRes* prop = drmModeGetProperty(lease_fd, props->props[i]);
        if (!prop) {
            continue;
        }
        if (std::strcmp(prop->name, "panel orientation") == 0) {
            for (int j = 0; j < prop->count_enums; j++) {
                if (prop->enums[j].value != props->prop_values[i]) {
                    continue;
                }
                const char* name = prop->enums[j].name;
                if (std::strcmp(name, "Normal") == 0) {
                    rotation = 0;
                } else if (std::strcmp(name, "Upside Down") == 0) {
                    rotation = 180;
                } else if (std::strcmp(name, "Left Side Up") == 0) {
                    rotation = 90;
                } else if (std::strcmp(name, "Right Side Up") == 0) {
                    rotation = 270;
                }
                break;
            }
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return rotation;
}

} // namespace

static std::unique_ptr<DrmLeaseClient> client;
static std::unique_ptr<TouchLog> touch_log;
static std::unique_ptr<DrmLeaseTouch> touch;

} // namespace Frontend

using namespace Frontend;

bool drm_lease_acquire(const char* connector, int rotation, const char* touch_device,
                       DrmLeaseOutput* out) {
    if (!drm_lease_dl_open()) {
        return false;
    }
    client = std::make_unique<DrmLeaseClient>();
    const bool explicit_connector = connector && connector[0] != '\0';
    if (!client->Acquire(explicit_connector ? connector : "", !explicit_connector)) {
        client.reset();
        return false;
    }

    drmModeModeInfo mode{};
    if (!ReadConnectorMode(client->GetLeaseFd(), client->GetConnectorId(), mode)) {
        LOG_ERROR(Frontend, "drm-lease: failed to read a mode from the leased connector");
        client.reset();
        return false;
    }
    out->width = mode.hdisplay;
    out->height = mode.vdisplay;
    LinkConnectorToLeasedCrtc(client->GetLeaseFd(), client->GetConnectorId(), mode);

    if (rotation < 0) {
        rotation = ReadPanelOrientation(client->GetLeaseFd(), client->GetConnectorId());
        if (rotation < 0) {
            rotation = 0;
            LOG_INFO(Frontend,
                     "drm-lease: connector reports no panel orientation, assuming unrotated; "
                     "set SECOND_SCREEN_ROTATION if the output is sideways");
        } else {
            LOG_INFO(Frontend, "drm-lease: panel orientation property gives rotation {}", rotation);
        }
    }
    out->fd = client->GetLeaseFd();
    out->connector_id = client->GetConnectorId();
    out->rotation = rotation;

    if (touch_device && touch_device[0] != '\0') {
        touch_log = std::make_unique<TouchLog>(out->width, out->height);
        touch = std::make_unique<DrmLeaseTouch>(*touch_log, touch_device);
    }

    LOG_INFO(Frontend, "drm-lease: secondary output ready, {}x{} rotation {}", out->width,
             out->height, rotation);
    return true;
}

bool drm_lease_offered(const char* connector) {
    if (!drm_lease_dl_open()) {
        return false;
    }
    DrmLeaseClient probe;
    const bool explicit_connector = connector && connector[0] != '\0';
    return probe.Acquire(explicit_connector ? connector : "", !explicit_connector, true);
}

bool drm_lease_alive(void) {
    if (!client) {
        return false;
    }
    client->PumpEvents();
    return !client->IsFinished();
}

void drm_lease_release(void) {
    touch.reset();
    touch_log.reset();
    client.reset();
}
