#pragma once
#include "light.h"
#include "material.h"
struct ray_counts {
    std::uint64_t paths = 0, shadows = 0;
};
class caustic_map;
struct transport_options {
    bool transparent_shadows = false;
    const caustic_map *caustics = nullptr;
};
struct environment {
    bool sky = true;
    color background{0, 0, 0};
    color radiance(const vec3 &d) const {
        if (!sky)
            return background;
        double a = 0.5 * (d.y() + 1);
        return (1 - a) * color(1, 1, 1) + a * color(0.5, 0.7, 1);
    }
};
color trace_path(ray r, const hittable &world, const light_list &lights, const environment &env,
                 int max_depth, sampler &rng, ray_counts &counts, transport_options options = {});
