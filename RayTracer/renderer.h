#pragma once
#include "framebuffer.h"
#include "scene.h"
#include <memory>

struct render_settings {
    int width = 640, samples = 64, max_depth = 16, threads = 0;
    std::uint64_t seed = 42;
    bool use_bvh = true;
    bool collect_guides = false, transparent_shadows = false;
    bool collect_features = false;
    std::string history_key;
    std::int64_t frame_index = -1;
    int caustic_photons = 0;
    double caustic_radius = 0.12;
    int height(double aspect) const;
    void validate(double aspect) const;
};
struct render_stats {
    int samples = 0, workers = 0, objects = 0;
    std::uint64_t path_rays = 0, shadow_rays = 0;
    std::uint64_t feature_rays = 0;
    double seconds = 0, build_seconds = 0;
    double photon_seconds = 0;
    std::uint64_t emitted_photons = 0, stored_photons = 0, photon_rays = 0;
};
struct render_result {
    frame_snapshot frame;
    render_stats stats;
};
// Owns immutable scene geometry and all worker lifetimes. Only complete sample
// passes are published; UI snapshots never access actively written pixels.
class render_session {
  public:
    render_session(scene content, render_settings settings);
    ~render_session();
    render_session(const render_session &) = delete;
    render_session &operator=(const render_session &) = delete;
    void cancel();
    void set_paused(bool paused);
    bool paused() const;
    bool done() const;
    void wait();
    frame_snapshot snapshot() const;
    render_stats stats() const;
    render_result capture() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
