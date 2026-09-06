#pragma once
#include "camera.h"
#include "color.h"
#include <filesystem>
#include <string>
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
    struct feature {
        color albedo, normal, variance;
        double depth = 0, coverage = 0, support = 0;
        point3 position;
    };
    std::vector<feature> features{};
    bool reconstructed = false;
    int comparison_view = 0; // 0 render, 1 reference, 2 absolute error.
    camera_settings camera{};
    std::string history_key{};
    std::vector<std::uint8_t> rgb(double exposure = 0) const;
};
void write_png(const std::filesystem::path &path, const frame_snapshot &frame, double exposure = 0);
// Linear scene-referred exports: exposure and tone mapping are deliberately absent.
void write_hdr(const std::filesystem::path &path, const frame_snapshot &frame);
void write_pfm(const std::filesystem::path &path, const frame_snapshot &frame);
frame_snapshot read_pfm(const std::filesystem::path &path);
void write_image(const std::filesystem::path &path, const frame_snapshot &frame,
                 double exposure = 0);
