// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
// Ported from Azahar-Enhanced (https://github.com/virtudude/azahar, branch drm-leasing) commit d16299140,
// src/citra_qt/drm_lease/drm_lease_touch.h. Edits for MHGU Recomp: EmuWindow replaced by TouchLog, which only logs
// the coordinates; std::thread and a flag in place of std::jthread, whose wait needs a newer libstdc++ than the
// Makefile's GLIBCXX_MAX.

#pragma once

#include <atomic>
#include <string>
#include <thread>

// The caller's side (src/backend/drm_lease.rs): a finger at panel pixel x, y (down false: lifted).
extern "C" void second_display_touch(bool down, unsigned x, unsigned y);

namespace Frontend {

/// Takes the place of Azahar's EmuWindow: hands each touch to second_display_touch (and logs where a finger goes down).
class TouchLog {
public:
    TouchLog(unsigned width_, unsigned height_) : width{width_}, height{height_} {}
    bool TouchPressed(unsigned x, unsigned y);
    void TouchMoved(unsigned x, unsigned y);
    void TouchReleased();
    struct Layout {
        unsigned width, height;
    };
    Layout GetFramebufferLayout() const {
        return {width, height};
    }

private:
    unsigned width, height;
};

/// Reads a touchscreen evdev device directly and feeds it to a TouchLog as
/// framebuffer-space touch input. Used for screens driven through a DRM lease,
/// where no windowing system delivers input for the panel. The device is
/// grabbed (EVIOCGRAB) so the compositor stops seeing its events.
class DrmLeaseTouch {
public:
    /// device_path: an evdev node path, an evdev device name, or "auto" to
    /// pick the only direct-touch device (preferring one named like the
    /// bottom screen when several exist).
    DrmLeaseTouch(TouchLog& window, std::string device_path);
    ~DrmLeaseTouch();

    DrmLeaseTouch(const DrmLeaseTouch&) = delete;
    DrmLeaseTouch& operator=(const DrmLeaseTouch&) = delete;

private:
    void Run();
    int OpenDevice();
    void ReadLoop(int fd);

    TouchLog& window;
    std::string device_path;
    bool swap_xy = false;
    bool invert_x = false;
    bool invert_y = false;
    std::atomic<bool> stop_requested{false};
    std::thread thread;
};

} // namespace Frontend
