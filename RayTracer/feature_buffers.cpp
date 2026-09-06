#include "feature_buffers.h"
#include "../third_party/json.hpp"
#include "denoiser.h"
#include <fstream>

std::vector<float> pack_reconstruction_input(const frame_snapshot &f) {
    auto size = f.linear.size();
    if (f.samples < 1 || f.features.size() != size || size != std::size_t(f.width) * f.height)
        throw std::invalid_argument("Reconstruction needs complete feature buffers");
    std::vector<float> out(size * 17);
    for (std::size_t i = 0; i < size; ++i) {
        const auto &g = f.features[i];
        const double values[17] = {
            f.linear[i].x(),        f.linear[i].y(), f.linear[i].z(), g.albedo.x(),
            g.albedo.y(),           g.albedo.z(),    g.normal.x(),    g.normal.y(),
            g.normal.z(),           g.depth,         g.coverage,      g.support,
            g.variance.x(),         g.variance.y(),  g.variance.z(),  double(f.samples),
            f.samples > 1 ? 1. : 0.};
        for (int c = 0; c < 17; ++c) {
            if (!std::isfinite(values[c]) || std::abs(values[c]) > 1e30)
                throw std::invalid_argument("Invalid neural input");
            out[std::size_t(c) * size + i] = static_cast<float>(values[c]);
        }
    }
    return out;
}
void write_feature_buffers(const std::filesystem::path &dir, const frame_snapshot &f) {
    pack_reconstruction_input(f); // Validate once before writing any buffers.
    std::filesystem::create_directories(dir);
    write_pfm(dir / "radiance.pfm", f);
    frame_snapshot buffer{f.width, f.height, f.samples, std::vector<color>(f.linear.size())};
    const char *names[] = {"albedo", "normal", "geometry", "variance", "position"};
    for (int channel = 0; channel < 5; ++channel) {
        for (std::size_t i = 0; i < f.linear.size(); ++i) {
            const auto &g = f.features[i];
            buffer.linear[i] = channel == 0   ? g.albedo
                               : channel == 1 ? g.normal
                               : channel == 2 ? color(g.depth, g.coverage, g.support)
                               : channel == 3 ? g.variance
                                              : g.position;
        }
        write_pfm(dir / (std::string(names[channel]) + ".pfm"), buffer);
    }
    write_pfm(dir / "atrous.pfm", denoise(f));
    nlohmann::json j = {{"schema_version", 1},
                        {"width", f.width},
                        {"height", f.height},
                        {"samples", f.samples},
                        {"channels", 17},
                        {"variance", "variance_of_mean; unavailable at samples=1"},
                        {"coordinates", "world normal/position; camera-ray distance"}};
    std::ofstream out(dir / "features.json");
    out << j.dump(2) << '\n';
    if (!out)
        throw std::runtime_error("Could not write feature manifest");
}
