#pragma once
#include "color.h"
#include <filesystem>
#include <vector>
struct surface_guide {
    vec3 normal;
    color albedo;
    double depth = 0;
    bool hit = false, filterable = false;
};
struct frame_snapshot {
    int width = 0, height = 0, samples = 0;
    std::vector<color> linear;
    std::vector<surface_guide> guides{};
    bool denoised = false;
    std::vector<std::uint8_t> rgb(double exposure = 0) const;
};
void write_png(const std::filesystem::path &path, const frame_snapshot &frame, double exposure = 0);
// Linear scene-referred exports: exposure and tone mapping are deliberately absent.
void write_hdr(const std::filesystem::path &path, const frame_snapshot &frame);
void write_pfm(const std::filesystem::path &path, const frame_snapshot &frame);
void write_image(const std::filesystem::path &path, const frame_snapshot &frame,
                 double exposure = 0);
