#include "feature_buffers.h"
#include "../third_party/json.hpp"
#include "denoiser.h"
#include <fstream>

std::vector<float> pack_reconstruction_input(const frame_snapshot &f, int schema) {
    std::vector<float> result;
    pack_reconstruction_input(f, result, schema);
    return result;
}
void pack_reconstruction_input(const frame_snapshot &f, std::vector<float> &out, int schema) {
    if (schema != 1 && schema != 2)
        throw std::invalid_argument("Unsupported neural feature schema");
    auto size = f.linear.size();
    if (f.samples < 1 || f.features.size() != size || size != std::size_t(f.width) * f.height)
        throw std::invalid_argument("Reconstruction needs complete feature buffers");
    const int channels = schema == 1 ? 17 : 27;
    out.resize(size * channels);
    for (std::size_t i = 0; i < size; ++i) {
        const auto &g = f.features[i];
        const int count = g.sample_count ? g.sample_count : f.samples;
        const double values[27] = {f.linear[i].x(),     f.linear[i].y(),
                                   f.linear[i].z(),     g.albedo.x(),
                                   g.albedo.y(),        g.albedo.z(),
                                   g.normal.x(),        g.normal.y(),
                                   g.normal.z(),        g.depth,
                                   g.coverage,          g.support,
                                   g.variance.x(),      g.variance.y(),
                                   g.variance.z(),      double(count),
                                   count > 1 ? 1. : 0., g.center_albedo.x(),
                                   g.center_albedo.y(), g.center_albedo.z(),
                                   g.center_normal.x(), g.center_normal.y(),
                                   g.center_normal.z(), g.center_depth,
                                   g.depth_variance,    g.normal_spread,
                                   g.center_support};
        for (int c = 0; c < channels; ++c) {
            if (!std::isfinite(values[c]) || std::abs(values[c]) > 1e30)
                throw std::invalid_argument("Invalid neural input");
            out[std::size_t(c) * size + i] = static_cast<float>(values[c]);
        }
    }
}
void write_feature_buffers(const std::filesystem::path &dir, const frame_snapshot &f) {
    pack_reconstruction_input(f); // Validate once before writing any buffers.
    std::filesystem::create_directories(dir);
    write_pfm(dir / "radiance.pfm", f);
    frame_snapshot buffer{f.width, f.height, f.samples, std::vector<color>(f.linear.size())};
    const char *names[] = {"albedo",        "normal",        "geometry", "variance", "position",
                           "center_albedo", "center_normal", "boundary", "sampling"};
    for (int channel = 0; channel < 9; ++channel) {
        for (std::size_t i = 0; i < f.linear.size(); ++i) {
            const auto &g = f.features[i];
            buffer.linear[i] =
                channel == 0   ? g.albedo
                : channel == 1 ? g.normal
                : channel == 2 ? color(g.depth, g.coverage, g.support)
                : channel == 3 ? g.variance
                : channel == 4 ? g.position
                : channel == 5 ? g.center_albedo
                : channel == 6 ? g.center_normal
                : channel == 7
                    ? color(g.center_depth, g.depth_variance, g.normal_spread)
                    : color(g.sample_count ? g.sample_count : f.samples,
                            (g.sample_count ? g.sample_count : f.samples) > 1, g.center_support);
        }
        write_pfm(dir / (std::string(names[channel]) + ".pfm"), buffer);
    }
    for (std::size_t i = 0; i < f.features.size(); ++i)
        buffer.linear[i] = color(f.features[i].thin_coverage, 0, 0);
    write_pfm(dir / "annotations.pfm", buffer);
    write_pfm(dir / "atrous.pfm", denoise(f));
    nlohmann::json j = {{"schema_version", 2},
                        {"width", f.width},
                        {"height", f.height},
                        {"samples", f.samples},
                        {"channels", 27},
                        {"variance", "variance_of_mean; unavailable at samples=1"},
                        {"coordinates", "world normal/position; camera-ray distance"}};
    std::ofstream out(dir / "features.json");
    out << j.dump(2) << '\n';
    if (!out)
        throw std::runtime_error("Could not write feature manifest");
}
