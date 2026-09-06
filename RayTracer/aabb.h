#pragma once
#include "interval.h"
#include "ray.h"
class aabb {
  public:
    point3 min{infinity, infinity, infinity}, max{-infinity, -infinity, -infinity};
    aabb() = default;
    aabb(const point3 &lo, const point3 &hi) : min(lo), max(hi) {}
    aabb(const aabb &a, const aabb &b) {
        for (int i = 0; i < 3; ++i) {
            min[i] = std::min(a.min[i], b.min[i]);
            max[i] = std::max(a.max[i], b.max[i]);
        }
    }
    bool hit(const ray &r, interval range) const {
        for (int i = 0; i < 3; ++i) {
            // Avoid 0 * infinity for parallel rays on slab boundaries.
            if (r.d()[i] == 0) {
                if (r.o()[i] < min[i] || r.o()[i] > max[i])
                    return false;
                continue;
            }
            double a = (min[i] - r.o()[i]) / r.d()[i], b = (max[i] - r.o()[i]) / r.d()[i];
            if (a > b)
                std::swap(a, b);
            range.min = std::max(range.min, a);
            range.max = std::min(range.max, b);
            if (range.max < range.min)
                return false;
        }
        return true;
    }
    int longest_axis() const {
        auto d = max - min;
        return d.x() > d.y() ? (d.x() > d.z() ? 0 : 2) : (d.y() > d.z() ? 1 : 2);
    }
};
