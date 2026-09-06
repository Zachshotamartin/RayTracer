#pragma once
#include "color.h"
#include "material.h"
#include "quad.h"
#include <vector>
struct light_sample {
    vec3 direction;
    double distance = infinity;
    color radiance;
    double pdf = 0;
    bool delta = true;
};
struct emitted_photon {
    ray path;
    color flux;
};
class light {
  public:
    virtual ~light() = default;
    virtual light_sample sample(const point3 &, sampler &) const = 0;
    virtual double pdf_for_hit(const point3 &, const hit_record &) const { return 0; }
    virtual emitted_photon emit(const aabb &, sampler &) const = 0;
    virtual const hittable *emitter() const { return nullptr; }
};
class PointLight : public light {
  public:
    PointLight(point3 position, color c, double intensity)
        : position_(position), power_(c * intensity) {}
    light_sample sample(const point3 &origin, sampler &) const override {
        vec3 d = position_ - origin;
        double distance = d.length();
        if (distance <= 1e-12)
            return {};
        return {d / distance, distance, power_ / (distance * distance), 1, true};
    }
    emitted_photon emit(const aabb &, sampler &rng) const override {
        return {ray(position_, random_unit_vector(rng)), power_ * (4 * rt_pi)};
    }

  private:
    point3 position_;
    color power_;
};
class DirectionalLight : public light {
  public:
    DirectionalLight(vec3 direction, color c, double intensity)
        : direction_(-to_unit_vector(direction)), radiance_(c * intensity) {}
    light_sample sample(const point3 &, sampler &) const override {
        return {direction_, infinity, radiance_, 1, true};
    }
    emitted_photon emit(const aabb &bounds, sampler &rng) const override {
        auto center = (bounds.min + bounds.max) / 2;
        double radius = (bounds.max - bounds.min).length() / 2 + 1e-4;
        auto disk = from_local(random_in_unit_disk(rng) * radius, direction_);
        return {ray(center + direction_ * (radius + 1e-4) + disk, -direction_),
                radiance_ * (rt_pi * radius * radius)};
    }

  private:
    vec3 direction_;
    color radiance_;
};
class AreaLight : public light {
  public:
    AreaLight(std::shared_ptr<quad> shape, color emission)
        : shape_(std::move(shape)), emission_(emission) {}
    light_sample sample(const point3 &origin, sampler &rng) const override {
        vec3 d = shape_->sample(rng) - origin;
        double distance = d.length();
        if (distance <= 1e-12)
            return {};
        vec3 direction = d / distance;
        double cosine = dot(shape_->normal(), -direction);
        if (cosine <= 0)
            return {};
        return {direction, distance, emission_, distance * distance / (cosine * shape_->area()),
                false};
    }
    double pdf_for_hit(const point3 &origin, const hit_record &rec) const override {
        if (rec.object != shape_.get() || !rec.front_face)
            return 0;
        vec3 d = rec.p - origin;
        double distance = d.length();
        if (distance <= 1e-12)
            return 0;
        double cosine = dot(shape_->normal(), -d / distance);
        return cosine > 0 ? d.dot() / (cosine * shape_->area()) : 0;
    }
    const hittable *emitter() const override { return shape_.get(); }
    emitted_photon emit(const aabb &, sampler &rng) const override {
        auto p = shape_->sample(rng);
        double u = rng.uniform(), phi = 2 * rt_pi * rng.uniform();
        auto d = from_local(
            vec3(std::sqrt(u) * std::cos(phi), std::sqrt(u) * std::sin(phi), std::sqrt(1 - u)),
            shape_->normal());
        return {ray(p + shape_->normal() * origin_epsilon(p), d),
                emission_ * (rt_pi * shape_->area())};
    }

  private:
    std::shared_ptr<quad> shape_;
    color emission_;
};
class light_list {
  public:
    void add(std::shared_ptr<light> item) { lights_.push_back(std::move(item)); }
    bool empty() const { return lights_.empty(); }
    bool owns_emitter(const hittable *object) const {
        for (const auto &item : lights_)
            if (item->emitter() == object)
                return true;
        return false;
    }
    emitted_photon emit(const aabb &bounds, sampler &rng) const {
        if (empty())
            return {};
        auto i = std::min(lights_.size() - 1, std::size_t(rng.uniform() * lights_.size()));
        auto photon = lights_[i]->emit(bounds, rng);
        photon.flux *= double(lights_.size());
        return photon;
    }
    // A straight connection approximation; refraction bending is handled by photon mapping.
    static color transmittance(const point3 &origin, const light_sample &sample,
                               const hittable &world, bool *crossed_glass = nullptr) {
        if (sample.pdf <= 0)
            return {};
        color transmission(1, 1, 1);
        double start = 1e-8;
        double end = std::isfinite(sample.distance) ? sample.distance - 4 * origin_epsilon(origin)
                                                    : infinity;
        for (int crossing = 0; crossing < 128; ++crossing) {
            hit_record rec;
            if (!world.hit(ray(origin, sample.direction), interval(start, end), rec))
                return transmission;
            transmission = transmission * rec.mat->shadow_transmission(sample.direction, rec);
            if (transmission.near_zero())
                return {};
            if (crossed_glass)
                *crossed_glass = true;
            start = rec.t + origin_epsilon(rec.p);
        }
        return {}; // Conservative termination for malformed or extremely layered geometry.
    }
    light_sample sample(const point3 &p, sampler &rng) const {
        if (empty())
            return {};
        auto i =
            std::min(lights_.size() - 1, static_cast<std::size_t>(rng.uniform() * lights_.size()));
        auto result = lights_[i]->sample(p, rng);
        result.pdf /= static_cast<double>(lights_.size());
        return result;
    }
    double pdf_for_hit(const point3 &p, const hit_record &rec) const {
        if (empty())
            return 0;
        double pdf = 0;
        for (const auto &light : lights_)
            pdf += light->pdf_for_hit(p, rec);
        return pdf / static_cast<double>(lights_.size());
    }
    static bool visible(const point3 &origin, const light_sample &sample, const hittable &world) {
        if (sample.pdf <= 0)
            return false;
        hit_record obstruction;
        // Only the finite segment before a point/area light can occlude it.
        double end = std::isfinite(sample.distance) ? sample.distance - 4 * origin_epsilon(origin)
                                                    : infinity;
        return end > 0 &&
               !world.hit(ray(origin, sample.direction), interval(1e-8, end), obstruction);
    }

  private:
    std::vector<std::shared_ptr<light>> lights_;
};
