#pragma once
#include "hittable.h"
#include <array>
// Oriented box with an orthonormal frame.
class cube : public hittable {
  public:
    cube(const point3 &center, const vec3 &lengths, const vec3 &u, const vec3 &v, const vec3 &w,
         std::shared_ptr<material> mat)
        : center_(center), half_(lengths / 2), axes_{u, v, w}, mat_(std::move(mat)) {
        for (int i = 0; i < 3; ++i) {
            if (!(lengths[i] > 0) || !std::isfinite(lengths[i]) ||
                std::fabs(axes_[i].dot() - 1) > 1e-8)
                throw std::invalid_argument("Box requires positive lengths and unit axes");
            for (int j = 0; j < i; ++j)
                if (std::fabs(dot(axes_[i], axes_[j])) > 1e-8)
                    throw std::invalid_argument("Box axes must be orthogonal");
        }
    }
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        double entry = -infinity, exit = infinity;
        vec3 entry_normal, exit_normal;
        for (int i = 0; i < 3; ++i) {
            double o = dot(r.o() - center_, axes_[i]), d = dot(r.d(), axes_[i]);
            if (d == 0) {
                if (o < -half_[i] || o > half_[i])
                    return false;
                continue;
            }
            double a = (-half_[i] - o) / d, b = (half_[i] - o) / d;
            vec3 na = -axes_[i], nb = axes_[i];
            if (a > b) {
                std::swap(a, b);
                std::swap(na, nb);
            }
            if (a > entry) {
                entry = a;
                entry_normal = na;
            }
            if (b < exit) {
                exit = b;
                exit_normal = nb;
            }
            if (entry > exit)
                return false;
        }
        bool entering = range.surrounds(entry);
        double t = entering ? entry : exit;
        if (!std::isfinite(t) || !range.surrounds(t))
            return false;
        rec.t = t;
        rec.p = r.at(t);
        rec.mat = mat_;
        rec.object = this;
        rec.set_face_normal(r, entering ? entry_normal : exit_normal);
        rec.u = dot(rec.p - center_, axes_[0]) / (2 * half_[0]) + 0.5;
        rec.v = dot(rec.p - center_, axes_[2]) / (2 * half_[2]) + 0.5;
        return true;
    }
    aabb bounding_box() const override {
        vec3 extent;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                extent[j] += std::fabs(axes_[i][j]) * half_[i];
        return aabb(center_ - extent, center_ + extent);
    }

  private:
    point3 center_;
    vec3 half_;
    std::array<vec3, 3> axes_;
    std::shared_ptr<material> mat_;
};
