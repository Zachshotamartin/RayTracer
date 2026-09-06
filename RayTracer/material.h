#pragma once
#include "hittable.h"
#include "texture.h"

struct scatter_sample {
    vec3 direction;
    color weight; // f * abs(n dot wi) / PDF, including branch probabilities
    double pdf = 0;
    bool delta = false;
};
inline vec3 from_local(const vec3 &p, const vec3 &normal) {
    vec3 tangent =
        to_unit_vector(cross(std::fabs(normal.x()) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0), normal));
    return p.x() * tangent + p.y() * cross(normal, tangent) + p.z() * normal;
}
class material {
  public:
    virtual ~material() = default;
    virtual bool sample(const vec3 &incoming, const hit_record &, sampler &,
                        scatter_sample &) const = 0;
    virtual color evaluate(const vec3 &, const vec3 &, const hit_record &) const { return {}; }
    virtual double pdf(const vec3 &, const vec3 &, const hit_record &) const { return 0; }
    virtual color emitted(const hit_record &) const { return {}; }
    virtual bool is_delta() const { return false; }
};
class lambertian : public material {
  public:
    explicit lambertian(color albedo) : albedo_(std::make_shared<solid_color>(albedo)) {}
    explicit lambertian(std::shared_ptr<texture> albedo) : albedo_(std::move(albedo)) {}
    bool sample(const vec3 &, const hit_record &rec, sampler &rng,
                scatter_sample &out) const override {
        double u = rng.uniform(), phi = 2 * rt_pi * rng.uniform();
        out.direction = from_local(
            vec3(std::sqrt(u) * std::cos(phi), std::sqrt(u) * std::sin(phi), std::sqrt(1 - u)),
            rec.normal);
        out.pdf = std::max(0.0, dot(rec.normal, out.direction)) / rt_pi;
        out.weight = albedo_->value(rec.u, rec.v, rec.p);
        out.delta = false;
        return out.pdf > 0;
    }
    color evaluate(const vec3 &, const vec3 &wi, const hit_record &rec) const override {
        return dot(rec.normal, wi) > 0 ? albedo_->value(rec.u, rec.v, rec.p) / rt_pi : color{};
    }
    double pdf(const vec3 &, const vec3 &wi, const hit_record &rec) const override {
        return std::max(0.0, dot(rec.normal, wi)) / rt_pi;
    }

  private:
    std::shared_ptr<texture> albedo_;
};
// Isotropic GGX conductor. A roughness of zero is an ideal mirror.
class metal : public material {
  public:
    metal(color reflectance, double roughness)
        : f0_(reflectance), roughness_(std::clamp(roughness, 0.0, 1.0)),
          alpha_(std::max(0.001, roughness_ * roughness_)) {}
    bool is_delta() const override { return roughness_ == 0; }
    bool sample(const vec3 &incoming, const hit_record &rec, sampler &rng,
                scatter_sample &out) const override {
        if (is_delta()) {
            out.direction = reflect(incoming, rec.normal);
            out.weight = fresnel(std::max(0.0, dot(-incoming, rec.normal)));
            out.pdf = 1;
            out.delta = true;
            return true;
        }
        double u = rng.uniform(), phi = 2 * rt_pi * rng.uniform();
        double cos_theta = std::sqrt((1 - u) / (1 + (alpha_ * alpha_ - 1) * u));
        double sin_theta = std::sqrt(std::max(0.0, 1 - cos_theta * cos_theta));
        vec3 h = from_local(vec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta),
                            rec.normal);
        if (dot(-incoming, h) <= 0)
            return false;
        out.direction = reflect(incoming, h);
        out.delta = false;
        out.pdf = pdf(incoming, out.direction, rec);
        if (out.pdf <= 0)
            return false;
        out.weight = evaluate(incoming, out.direction, rec) *
                     (std::max(0.0, dot(rec.normal, out.direction)) / out.pdf);
        return true;
    }
    color evaluate(const vec3 &incoming, const vec3 &wi, const hit_record &rec) const override {
        double no = dot(rec.normal, -incoming), ni = dot(rec.normal, wi);
        if (is_delta() || no <= 0 || ni <= 0 || (wi - incoming).near_zero())
            return {};
        vec3 h = to_unit_vector(wi - incoming);
        return fresnel(std::max(0.0, dot(-incoming, h))) *
               (distribution(dot(rec.normal, h)) * mask(no) * mask(ni) / (4 * no * ni));
    }
    double pdf(const vec3 &incoming, const vec3 &wi, const hit_record &rec) const override {
        if (is_delta() || dot(rec.normal, wi) <= 0 || dot(rec.normal, -incoming) <= 0 ||
            (wi - incoming).near_zero())
            return 0;
        vec3 h = to_unit_vector(wi - incoming);
        double oh = dot(-incoming, h), nh = dot(rec.normal, h);
        return oh > 0 ? distribution(nh) * std::max(0.0, nh) / (4 * oh) : 0;
    }

  private:
    color fresnel(double cosine) const {
        return f0_ + (color(1, 1, 1) - f0_) * std::pow(1 - std::clamp(cosine, 0.0, 1.0), 5);
    }
    double distribution(double nh) const {
        if (nh <= 0)
            return 0;
        double a2 = alpha_ * alpha_, d = nh * nh * (a2 - 1) + 1;
        return a2 / (rt_pi * d * d);
    }
    double mask(double n) const {
        return 2 * n / (n + std::sqrt(alpha_ * alpha_ + (1 - alpha_ * alpha_) * n * n));
    }
    color f0_;
    double roughness_, alpha_;
};
class dielectric : public material {
  public:
    explicit dielectric(double index) : index_(index) {
        if (!(index > 0) || !std::isfinite(index))
            throw std::invalid_argument("Refractive index must be positive and finite");
    }
    bool is_delta() const override { return true; }
    bool sample(const vec3 &incoming, const hit_record &rec, sampler &rng,
                scatter_sample &out) const override {
        double eta = rec.front_face ? 1 / index_ : index_;
        double cosine = std::clamp(dot(-incoming, rec.normal), 0.0, 1.0);
        double sine = std::sqrt(std::max(0.0, 1 - cosine * cosine));
        bool reflected = eta * sine > 1 || reflectance(cosine, eta) > rng.uniform();
        out.direction =
            reflected ? reflect(incoming, rec.normal) : refract(incoming, rec.normal, eta);
        out.weight = reflected ? color(1, 1, 1) : color(eta * eta, eta * eta, eta * eta);
        out.pdf = 1;
        out.delta = true;
        return true;
    }
    static double reflectance(double cosine, double index) {
        double r = (1 - index) / (1 + index);
        r *= r;
        return r + (1 - r) * std::pow(1 - cosine, 5);
    }

  private:
    double index_;
};
class diffuse_light : public material {
  public:
    explicit diffuse_light(color emission) : emission_(emission) {}
    bool sample(const vec3 &, const hit_record &, sampler &, scatter_sample &) const override {
        return false;
    }
    color emitted(const hit_record &rec) const override {
        return rec.front_face ? emission_ : color{};
    }

  private:
    color emission_;
};
