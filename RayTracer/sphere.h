#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"

class sphere : public hittable {
  public:
    sphere(const point3 &center, double radius, std::shared_ptr<material> mat)
        : center(center), radius(radius), mat(std::move(mat)) {
        if (!(radius > 0) || !std::isfinite(radius))
            throw std::invalid_argument("Sphere radius must be positive and finite");
    }
    aabb bounding_box() const override {
        vec3 r(radius, radius, radius);
        return aabb(center - r, center + r);
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        vec3 oc = center - r.o();
        auto a = r.d().dot();
        if (a == 0)
            return false;
        auto b = dot(r.d(), oc);
        auto c = oc.dot() - radius * radius;

        auto discr = discriminant(a, b, c);
        if (discr < 0)
            return false;

        auto sqrtd = std::sqrt(discr);

        // Find the nearest root that lies in the acceptable range.
        auto root = (b - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (b + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        rec.object = this;
        rec.u = (std::atan2(-outward_normal.z(), outward_normal.x()) + rt_pi) / (2 * rt_pi);
        rec.v = std::acos(std::clamp(-outward_normal.y(), -1.0, 1.0)) / rt_pi;
        return true;
    }

  private:
    point3 center;
    double radius;
    std::shared_ptr<material> mat;
};

#endif
