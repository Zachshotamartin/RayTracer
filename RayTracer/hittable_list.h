#pragma once
#include "hittable.h"
#include <vector>
class hittable_list : public hittable {
  public:
    std::vector<std::shared_ptr<hittable>> objects;
    void clear() { objects.clear(); }
    void add(std::shared_ptr<hittable> object) { objects.push_back(std::move(object)); }
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        bool found = false;
        hit_record candidate;
        for (const auto &object : objects)
            if (object->hit(r, range, candidate)) {
                found = true;
                range.max = candidate.t;
                rec = candidate;
            }
        return found;
    }
    aabb bounding_box() const override {
        aabb result;
        for (const auto &object : objects)
            result = aabb(result, object->bounding_box());
        return result;
    }
};
