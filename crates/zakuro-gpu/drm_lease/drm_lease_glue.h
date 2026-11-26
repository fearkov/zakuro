// What Azahar's files take from its common/ headers: its integer names and its LOG_ macros, whose "{}" fields are
// filled in here. New glue for MHGU Recomp (MIT, as the rest of runtime/).
#pragma once
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string_view>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using s32 = std::int32_t;

inline void drm_lease_log_fields(std::ostringstream& out, std::string_view format) {
    out << format;
}

template <typename T, typename... Rest>
void drm_lease_log_fields(std::ostringstream& out, std::string_view format, const T& value, const Rest&... rest) {
    const auto field = format.find("{}");
    if (field == std::string_view::npos) {
        out << format;
        return;
    }
    out << format.substr(0, field) << value;
    drm_lease_log_fields(out, format.substr(field + 2), rest...);
}

template <typename... Args>
void drm_lease_log(std::string_view format, const Args&... args) {
    std::ostringstream out;
    drm_lease_log_fields(out, format, args...);
    fprintf(stderr, "[second-display] %s\n", out.str().c_str());
}

#define LOG_INFO(category, ...) drm_lease_log(__VA_ARGS__)
#define LOG_WARNING(category, ...) drm_lease_log(__VA_ARGS__)
#define LOG_ERROR(category, ...) drm_lease_log(__VA_ARGS__)
