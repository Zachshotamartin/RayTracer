#pragma once
#include "hittable.h"
#include <array>

class triangle : public hittable {
  public:
    triangle(std::array<point3, 3> points, std::shared_ptr<material> mat,
             std::array<vec3, 3> uv = {}, std::array<vec3, 3> normals = {})
        : p_(points), uv_(uv), normals_(normals), mat_(std::move(mat)) {
        for (auto p : p_)
            for (int c = 0; c < 3; ++c)
                if (!std::isfinite(p[c]))
                    throw std::invalid_argument("Non-finite triangle vertex");
        normal_ = to_unit_vector(cross(p_[1] - p_[0], p_[2] - p_[0]));
        for (auto p : p_)
            bounds_ = aabb(bounds_, aabb(p, p));
        double pad = 1e-6 * std::max(1.0, (bounds_.max - bounds_.min).length());
        bounds_ = aabb(bounds_.min - vec3(pad, pad, pad), bounds_.max + vec3(pad, pad, pad));
    }
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        // Double-precision, two-sided Moller-Trumbore intersection.
        vec3 e1 = p_[1] - p_[0], e2 = p_[2] - p_[0], h = cross(r.d(), e2);
        double det = dot(e1, h);
        if (std::fabs(det) < 1e-14 * e1.length() * e2.length() * r.d().length())
            return false;
        if (det == 0)
            return false;
        vec3 s = r.o() - p_[0];
        double u = dot(s, h) / det;
        if (u < 0 || u > 1)
            return false;
        vec3 q = cross(s, e1);
        double v = dot(r.d(), q) / det;
        if (v < 0 || u + v > 1)
            return false;
        double t = dot(e2, q) / det;
        if (!range.surrounds(t))
            return false;
        rec.t = t;
        rec.p = r.at(t);
        rec.mat = mat_;
        rec.object = this;
        rec.set_face_normal(r, normal_);
        vec3 uv = (1 - u - v) * uv_[0] + u * uv_[1] + v * uv_[2];
        rec.u = uv.x();
        rec.v = uv.y();
        if (!normals_[0].near_zero() && !normals_[1].near_zero() && !normals_[2].near_zero()) {
            vec3 n = (1 - u - v) * normals_[0] + u * normals_[1] + v * normals_[2];
            if (!n.near_zero()) {
                n = to_unit_vector(n);
                if (dot(n, rec.geometric_normal) < 0)
                    n = -n;
                if (dot(n, r.d()) < 0)
                    rec.normal = n;
            }
        }
        return true;
    }
    aabb bounding_box() const override { return bounds_; }

  private:
    std::array<point3, 3> p_;
    std::array<vec3, 3> uv_, normals_;
    std::shared_ptr<material> mat_;
    vec3 normal_;
    aabb bounds_;
};
