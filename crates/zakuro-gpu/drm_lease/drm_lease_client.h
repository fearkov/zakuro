// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
// Ported from Azahar-Enhanced (https://github.com/virtudude/azahar, branch drm-leasing) commit d16299140,
// src/citra_qt/drm_lease/drm_lease_client.h. Edits for MHGU Recomp: include paths, IsFinished(), a timeout on the first roundtrip, Acquire's offer_only.

#pragma once

#include <functional>
#include <string>
#include <vector>
#include "drm_lease_glue.h"

struct wl_display;
struct wl_registry;
struct wp_drm_lease_device_v1;
struct wp_drm_lease_connector_v1;
struct wp_drm_lease_v1;

namespace Frontend {

/// Client for the drm-lease-v1 Wayland protocol. Connects to the compositor
/// on a dedicated wl_display, leases a connector and hands out the lease fd.
class DrmLeaseClient {
public:
    DrmLeaseClient();
    ~DrmLeaseClient();

    DrmLeaseClient(const DrmLeaseClient&) = delete;
    DrmLeaseClient& operator=(const DrmLeaseClient&) = delete;

    /// Leases a connector from the compositor. connector_name selects a
    /// specific offer; when empty, the first eligible offer is used, where
    /// internal_only restricts eligibility to internal panel types
    /// (DSI/eDP/LVDS/DPI) so auto-detection cannot grab an external display
    /// or VR headset. Blocks until the compositor answers. offer_only (MHGU
    /// Recomp): true once such a connector is offered, without leasing it.
    bool Acquire(const std::string& connector_name, bool internal_only, bool offer_only = false);

    /// Non-blocking dispatch so compositor events (notably finished) are
    /// observed while the lease is held. Call regularly.
    void PumpEvents();

private:
    /// Dispatches until condition() holds or timeout_ms elapses. Returns the
    /// final condition state; false also on connection failure.
    bool DispatchUntil(const std::function<bool()>& condition, int timeout_ms);

public:
    int GetLeaseFd() const {
        return lease_fd;
    }

    bool IsFinished() const {
        return lease_finished || pump_dead;
    }

    u32 GetConnectorId() const {
        return leased_connector_id;
    }

    struct ConnectorOffer {
        wp_drm_lease_connector_v1* handle = nullptr;
        std::string name;
        std::string description;
        u32 connector_id = 0;
        bool done = false;
    };

private:
    friend struct DrmLeaseClientCallbacks;

    wl_display* display = nullptr;
    wl_registry* registry = nullptr;
    wp_drm_lease_device_v1* device = nullptr;
    wp_drm_lease_v1* lease = nullptr;

    std::vector<ConnectorOffer> connectors;
    int enum_fd = -1;
    int lease_fd = -1;
    u32 leased_connector_id = 0;
    bool device_done = false;
    bool lease_finished = false;
    bool pump_dead = false;
};

} // namespace Frontend
