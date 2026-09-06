#pragma once
#include "color.h"
#include <filesystem>
#include <vector>
struct frame_snapshot {
    int width = 0, height = 0, samples = 0;
    std::vector<color> linear;
    std::vector<std::uint8_t> rgb(double exposure = 0) const;
};
void write_png(const std::filesystem::path &path, const frame_snapshot &frame, double exposure = 0);
