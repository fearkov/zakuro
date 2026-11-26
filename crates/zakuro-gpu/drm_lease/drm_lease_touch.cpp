// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
// Ported from Azahar-Enhanced (https://github.com/virtudude/azahar, branch drm-leasing) commit d16299140,
// src/citra_qt/drm_lease/drm_lease_touch.cpp. Edits for MHGU Recomp: TouchLog in place of EmuWindow (it hands the touches to second_display_touch), a log line
// for each node that can't be opened or isn't direct touch, nodes opened with the openat system call (OpenInputNode), MHGU_SECOND_DISPLAY_ names in place of AZAHAR_DRM_LEASE_ (this repository: SECOND_SCREEN_).

#include <algorithm>
#include <sys/syscall.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "drm_lease_glue.h"
#include "drm_lease_touch.h"

namespace Frontend {

namespace {

// (MHGU Recomp) An input node opened with the system call itself, not libc's open: started from Steam, the game has
// Steam's overlay library preloaded, whose open wrapper answers ENODEV for /dev/input nodes (Thor, 2026-11-21: the
// touchscreens failed with "No such device" in the game while the same open worked from a shell).
int OpenInputNode(const char* path) {
    return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, O_RDONLY | O_NONBLOCK | O_CLOEXEC));
}

bool HasBit(const u8* bits, unsigned bit) {
    return bits[bit / 8] & (1 << (bit % 8));
}

bool IsDirectTouchDevice(int fd) {
    u8 props[INPUT_PROP_MAX / 8 + 1]{};
    u8 abs_bits[ABS_MAX / 8 + 1]{};
    if (ioctl(fd, EVIOCGPROP(sizeof(props)), props) < 0 ||
        ioctl(fd, EVIOCGBIT(EV_ABS, sizeof(abs_bits)), abs_bits) < 0) {
        return false;
    }
    return HasBit(props, INPUT_PROP_DIRECT) && HasBit(abs_bits, ABS_MT_POSITION_X);
}

// name set: return the direct-touch device with that evdev name. name
// empty: return the only direct-touch device; with several, a unique
// device named like the bottom screen wins, otherwise refuse to guess.
std::optional<std::string> FindDirectTouchDevice(const std::string& name) {
    std::vector<std::pair<std::string, std::string>> candidates; // path, evdev name
    DIR* dir = opendir("/dev/input");
    if (!dir) {
        return std::nullopt;
    }
    while (dirent* entry = readdir(dir)) {
        if (std::strncmp(entry->d_name, "event", 5) != 0) {
            continue;
        }
        const std::string path = std::string("/dev/input/") + entry->d_name;
        const int fd = OpenInputNode(path.c_str());
        if (fd < 0) {
            // (MHGU Recomp: why a node can't be read, once per node, for finding the touchscreen on a new device)
            const int err = errno;
            static std::vector<std::string> told;
            if (std::find(told.begin(), told.end(), path) == told.end()) {
                told.push_back(path);
                LOG_INFO(Frontend, "drm-lease touch: cannot open {}: {}", path, std::strerror(err));
            }
            continue;
        }
        char device_name[80]{};
        ioctl(fd, EVIOCGNAME(sizeof(device_name) - 1), device_name);
        if (IsDirectTouchDevice(fd)) {
            candidates.emplace_back(path, device_name);
        } else if (std::strstr(device_name, "touch")) {
            static std::vector<std::string> told;
            if (std::find(told.begin(), told.end(), path) == told.end()) {
                told.push_back(path);
                LOG_INFO(Frontend, "drm-lease touch: {} ('{}') is not a direct-touch device", path, device_name);
            }
        }
        close(fd);
    }
    closedir(dir);

    if (!name.empty()) {
        for (const auto& [path, dev_name] : candidates) {
            if (dev_name == name) {
                return path;
            }
        }
        return std::nullopt;
    }
    if (candidates.size() == 1) {
        return candidates.front().first;
    }
    std::optional<std::string> bottom;
    for (const auto& [path, dev_name] : candidates) {
        if (dev_name.find("bottom") != std::string::npos) {
            if (bottom) {
                bottom.reset();
                break;
            }
            bottom = path;
        }
    }
    if (bottom) {
        return bottom;
    }
    LOG_ERROR(Frontend,
              "drm-lease touch: {} touch devices found and none is unambiguous, set "
              "SECOND_SCREEN_TOUCH explicitly",
              candidates.size());
    return std::nullopt;
}

} // namespace

namespace {
bool EnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}
} // namespace

bool TouchLog::TouchPressed(unsigned x, unsigned y) {
    LOG_INFO(Frontend, "drm-lease touch: down {} {}", x, y);
    second_display_touch(true, x, y);
    return true;
}

void TouchLog::TouchMoved(unsigned x, unsigned y) {
    second_display_touch(true, x, y);
}

void TouchLog::TouchReleased() {
    second_display_touch(false, 0, 0);
}

DrmLeaseTouch::DrmLeaseTouch(TouchLog& window_, std::string device_path_)
    : window{window_}, device_path{std::move(device_path_)} {
    // Digitizer orientation quirks: swap is applied before inversion.
    swap_xy = EnvFlag("SECOND_SCREEN_TOUCH_SWAP_XY");
    invert_x = EnvFlag("SECOND_SCREEN_TOUCH_INVERT_X");
    invert_y = EnvFlag("SECOND_SCREEN_TOUCH_INVERT_Y");
    thread = std::thread([this] { Run(); });
}

DrmLeaseTouch::~DrmLeaseTouch() {
    stop_requested = true;
    thread.join();
}

int DrmLeaseTouch::OpenDevice() {
    std::string path = device_path;
    if (path.empty() || path[0] != '/') {
        const auto found = FindDirectTouchDevice(path == "auto" ? "" : path);
        if (!found) {
            return -1;
        }
        path = *found;
    }

    const int fd = OpenInputNode(path.c_str());
    if (fd < 0) {
        return -1;
    }

    if (ioctl(fd, EVIOCGRAB, 1) < 0) {
        LOG_ERROR(Frontend, "drm-lease touch: EVIOCGRAB on {} failed: {}", path,
                  std::strerror(errno));
        close(fd);
        return -1;
    }

    char name[64]{};
    ioctl(fd, EVIOCGNAME(sizeof(name) - 1), name);
    LOG_INFO(Frontend, "drm-lease touch: grabbed {} ('{}')", path, name);
    return fd;
}

void DrmLeaseTouch::Run() {
    while (!stop_requested) {
        const int fd = OpenDevice();
        if (fd < 0) {
            // Device missing or not yet powered (e.g. across suspend); retry.
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        ReadLoop(fd);
        ioctl(fd, EVIOCGRAB, 0);
        close(fd);
        window.TouchReleased();
    }
}

void DrmLeaseTouch::ReadLoop(int fd) {
    input_absinfo abs_x{};
    input_absinfo abs_y{};
    if (ioctl(fd, EVIOCGABS(ABS_MT_POSITION_X), &abs_x) < 0 ||
        ioctl(fd, EVIOCGABS(ABS_MT_POSITION_Y), &abs_y) < 0) {
        LOG_ERROR(Frontend, "drm-lease touch: failed to read axis ranges");
        return;
    }
    const float range_x = static_cast<float>(abs_x.maximum - abs_x.minimum);
    const float range_y = static_cast<float>(abs_y.maximum - abs_y.minimum);
    if (range_x <= 0.0f || range_y <= 0.0f) {
        LOG_ERROR(Frontend, "drm-lease touch: degenerate axis ranges");
        return;
    }

    // Multitouch protocol B, slot 0 only: the 3DS touchscreen is single-touch.
    int slot = 0;
    int raw_x = abs_x.value;
    int raw_y = abs_y.value;
    int tracking_id = -1;
    bool touching = false;

    while (!stop_requested) {
        pollfd pfd{.fd = fd, .events = POLLIN, .revents = 0};
        const int ret = poll(&pfd, 1, 500);
        if (ret < 0 && errno != EINTR) {
            return;
        }
        if (ret <= 0 || !(pfd.revents & POLLIN)) {
            if (pfd.revents & (POLLERR | POLLHUP)) {
                return;
            }
            continue;
        }

        input_event events[64];
        const ssize_t bytes = read(fd, events, sizeof(events));
        if (bytes < 0) {
            if (errno == EAGAIN || errno == EINTR) {
                continue;
            }
            LOG_WARNING(Frontend, "drm-lease touch: read failed: {}", std::strerror(errno));
            return;
        }

        const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(input_event);
        for (std::size_t i = 0; i < count; i++) {
            const input_event& ev = events[i];
            switch (ev.type) {
            case EV_ABS:
                switch (ev.code) {
                case ABS_MT_SLOT:
                    slot = ev.value;
                    break;
                case ABS_MT_TRACKING_ID:
                    if (slot == 0) {
                        tracking_id = ev.value;
                    }
                    break;
                case ABS_MT_POSITION_X:
                    if (slot == 0) {
                        raw_x = ev.value;
                    }
                    break;
                case ABS_MT_POSITION_Y:
                    if (slot == 0) {
                        raw_y = ev.value;
                    }
                    break;
                }
                break;
            case EV_SYN: {
                if (ev.code != SYN_REPORT) {
                    break;
                }
                const bool now_touching = tracking_id >= 0;
                const auto& layout = window.GetFramebufferLayout();
                float nx = static_cast<float>(raw_x - abs_x.minimum) / range_x;
                float ny = static_cast<float>(raw_y - abs_y.minimum) / range_y;
                if (swap_xy) {
                    std::swap(nx, ny);
                }
                if (invert_x) {
                    nx = 1.0f - nx;
                }
                if (invert_y) {
                    ny = 1.0f - ny;
                }
                const auto fb_x = static_cast<unsigned>(nx * static_cast<float>(layout.width));
                const auto fb_y = static_cast<unsigned>(ny * static_cast<float>(layout.height));

                if (now_touching && !touching) {
                    touching = window.TouchPressed(fb_x, fb_y);
                } else if (now_touching && touching) {
                    window.TouchMoved(fb_x, fb_y);
                } else if (!now_touching && touching) {
                    window.TouchReleased();
                    touching = false;
                }
                break;
            }
            }
        }
    }
}

} // namespace Frontend
