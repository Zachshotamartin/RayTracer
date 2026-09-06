#include "caustics.h"

caustic_map::cell caustic_map::key(const point3 &p) const {
    return {std::int64_t(std::floor(p.x() / radius_)), std::int64_t(std::floor(p.y() / radius_)),
            std::int64_t(std::floor(p.z() / radius_))};
}
caustic_map::caustic_map(const hittable &world, const light_list &lights, int count, int depth,
                         double radius, std::uint64_t seed, const std::atomic<bool> *cancel)
    : radius_(radius) {
    if (count < 1 || depth < 1 || !(radius > 0) || !std::isfinite(radius))
        throw std::invalid_argument("Invalid caustic photon settings");
    if (lights.empty())
        return;
    auto bounds = world.bounding_box();
    // Bound coordinate-to-grid conversion before converting floating point to integers.
    for (int c = 0; c < 3; ++c)
        if (!std::isfinite(bounds.min[c]) || !std::isfinite(bounds.max[c]) ||
            std::max(std::fabs(bounds.min[c]), std::fabs(bounds.max[c])) / radius > 1e15)
            throw std::invalid_argument("Scene extent is too large for this photon radius");
    for (int i = 0; i < count; ++i) {
        if (cancel && cancel->load())
            break;
        auto rng = sampler::for_pixel(seed ^ 0x70686f746f6eULL, std::uint64_t(i), 0);
        auto photon = lights.emit(bounds, rng);
        color flux = photon.flux / count;
        auto r = photon.path;
        ++emitted;
        int specular = 0;
        for (int bounce = 0; bounce < depth; ++bounce) {
            ++traced_rays;
            hit_record rec;
            if (!world.hit(r, interval(1e-8, infinity), rec))
                break;
            if (!rec.mat->is_delta()) {
                if (specular > 0 && rec.mat->is_diffuse()) {
                    auto index = photons_.size();
                    photons_.push_back({{rec.p, -r.d(), rec.geometric_normal, flux}, specular});
                    cells_[key(rec.p)].push_back(index);
                }
                // Only L S+ D paths belong to this caustic map.
                break;
            }
            scatter_sample scattered;
            if (!rec.mat->sample_photon(r.d(), rec, rng, scattered))
                break;
            flux = flux * scattered.weight;
            if (flux.near_zero())
                break;
            ++specular;
            r = ray(offset_origin(rec, scattered.direction), to_unit_vector(scattered.direction));
        }
    }
}
color caustic_map::radiance(const vec3 &incoming, const hit_record &rec,
                            int remaining_depth) const {
    if (!rec.mat->is_diffuse())
        return {};
    auto [x, y, z] = key(rec.p);
    auto normal = rec.geometric_normal.near_zero() ? rec.normal : rec.geometric_normal;
    color sum;
    const double r2 = radius_ * radius_;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                auto found = cells_.find({x + dx, y + dy, z + dz});
                if (found == cells_.end())
                    continue;
                for (auto index : found->second) {
                    const auto &stored = photons_[index];
                    const auto &p = stored.value;
                    auto offset = p.position - rec.p;
                    double distance2 = offset.dot();
                    // The light endpoint and specular vertices must fit the path budget.
                    if (stored.specular_vertices + 1 > remaining_depth || distance2 >= r2 ||
                        dot(normal, p.normal) < 0.9 ||
                        std::fabs(dot(offset, normal)) > radius_ * 0.05)
                        continue;
                    double kernel = 2 * (1 - distance2 / r2) / (rt_pi * r2);
                    sum += rec.mat->evaluate(incoming, p.direction, rec) * p.flux * kernel;
                }
            }
    return sum;
}
