#pragma once
#include "hittable_list.h"
class bvh_node : public hittable {
  public:
    explicit bvh_node(std::vector<std::shared_ptr<hittable>> objects)
        : bvh_node(objects, 0, objects.size()) {}
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        if (!box_.hit(r, range))
            return false;
        bool left_hit = left_->hit(r, range, rec);
        if (left_hit)
            range.max = rec.t;
        bool right_hit = right_ && right_->hit(r, range, rec);
        return left_hit || right_hit;
    }
    aabb bounding_box() const override { return box_; }

  private:
    bvh_node(std::vector<std::shared_ptr<hittable>> &objects, std::size_t begin, std::size_t end) {
        if (begin == end)
            throw std::invalid_argument("Cannot build an empty BVH");
        for (auto i = begin; i < end; ++i)
            box_ = aabb(box_, objects[i]->bounding_box());
        if (end - begin == 1) {
            left_ = objects[begin];
            return;
        }
        int axis = box_.longest_axis();
        std::stable_sort(objects.begin() + begin, objects.begin() + end,
                         [axis](const auto &a, const auto &b) {
                             auto aa = a->bounding_box(), bb = b->bounding_box();
                             return aa.min[axis] + aa.max[axis] < bb.min[axis] + bb.max[axis];
                         });
        auto mid = begin + (end - begin) / 2;
        left_ = std::shared_ptr<hittable>(new bvh_node(objects, begin, mid));
        right_ = std::shared_ptr<hittable>(new bvh_node(objects, mid, end));
    }
    std::shared_ptr<hittable> left_, right_;
    aabb box_;
};
