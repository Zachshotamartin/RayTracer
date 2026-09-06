#pragma once
#include "hittable.h"
class quad : public hittable {
  public:
    quad(point3 corner, vec3 u, vec3 v, std::shared_ptr<material> mat)
        : corner_(corner), u_(u), v_(v), mat_(std::move(mat)) {
        vec3 n = cross(u, v);
        area_ = n.length();
        if (!(area_ > 0))
            throw std::invalid_argument("Quad must have nonzero area");
        normal_ = n / area_;
        inverse_cross_ = n / n.dot();
    }
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        double d = dot(normal_, r.d());
        if (std::fabs(d) < 1e-12)
            return false;
        double t = dot(corner_ - r.o(), normal_) / d;
        if (!range.surrounds(t))
            return false;
        vec3 p = r.at(t) - corner_;
        double u = dot(inverse_cross_, cross(p, v_)), v = dot(inverse_cross_, cross(u_, p));
        if (u < 0 || u > 1 || v < 0 || v > 1)
            return false;
        rec.t = t;
        rec.p = r.at(t);
        rec.u = u;
        rec.v = v;
        rec.mat = mat_;
        rec.object = this;
        rec.set_face_normal(r, normal_);
        return true;
    }
    point3 sample(sampler &rng) const {
        double u = rng.uniform(), v = rng.uniform();
        return corner_ + u * u_ + v * v_;
    }
    const vec3 &normal() const { return normal_; }
    double area() const { return area_; }
    aabb bounding_box() const override {
        aabb result;
        for (auto p : {corner_, corner_ + u_, corner_ + v_, corner_ + u_ + v_})
            result = aabb(result, aabb(p, p));
        vec3 pad(1e-6, 1e-6, 1e-6);
        return aabb(result.min - pad, result.max + pad);
    }

  private:
    point3 corner_;
    vec3 u_, v_, normal_, inverse_cross_;
    double area_;
    std::shared_ptr<material> mat_;
};
