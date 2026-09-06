#pragma once
#include "aabb.h"
#include <memory>
class material;
class hittable;
struct hit_record {
    point3 p;
    vec3 normal;
    vec3 geometric_normal;
    double t = 0, u = 0, v = 0;
    bool front_face = true;
    std::shared_ptr<material> mat;
    const hittable *object = nullptr;
    void set_face_normal(const ray &r, const vec3 &outward) {
        front_face = dot(r.d(), outward) < 0;
        normal = front_face ? outward : -outward;
        geometric_normal = normal;
    }
};
class hittable {
  public:
    virtual ~hittable() = default;
    virtual bool hit(const ray &, interval, hit_record &) const = 0;
    virtual aabb bounding_box() const = 0;
};
inline double origin_epsilon(const point3 &p) {
    return 1e-6 * std::max({1.0, std::fabs(p.x()), std::fabs(p.y()), std::fabs(p.z())});
}
inline point3 offset_origin(const hit_record &rec, const vec3 &direction) {
    auto normal = rec.geometric_normal.near_zero() ? rec.normal : rec.geometric_normal;
    return rec.p + (dot(direction, normal) >= 0 ? 1 : -1) * origin_epsilon(rec.p) * normal;
}
