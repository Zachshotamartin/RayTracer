#pragma once
#include "light.h"
#include <atomic>
#include <map>
#include <tuple>
struct caustic_photon {
    point3 position;
    vec3 direction, normal;
    color flux;
};
class caustic_map {
  public:
    caustic_map(const hittable &world, const light_list &lights, int count, int depth,
                double radius, std::uint64_t seed, const std::atomic<bool> *cancel = nullptr);
    color radiance(const vec3 &incoming, const hit_record &rec, int remaining_depth = 128) const;
    std::size_t stored() const { return photons_.size(); }
    std::uint64_t emitted = 0, traced_rays = 0;

  private:
    using cell = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    struct stored_photon {
        caustic_photon value;
        int specular_vertices;
    };
    cell key(const point3 &p) const;
    double radius_;
    std::vector<stored_photon> photons_;
    std::map<cell, std::vector<std::size_t>> cells_;
};
