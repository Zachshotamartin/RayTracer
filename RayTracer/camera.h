#pragma once
#include "ray.h"
struct camera_settings {
    double aspect_ratio = 16.0 / 9.0, vfov = 40, defocus_angle = 0, focus_dist = 10;
    point3 lookfrom{7, 3, 7}, lookat{0, 1, 0};
    vec3 vup{0, 1, 0};
};
class camera {
  public:
    camera(const camera_settings &settings, int width, int height) : center_(settings.lookfrom) {
        if (width < 1 || height < 1 || !(settings.vfov > 0 && settings.vfov < 179) ||
            !(settings.focus_dist > 0) || settings.defocus_angle < 0 ||
            settings.defocus_angle >= 179)
            throw std::invalid_argument(
                "Invalid camera dimensions, field of view, focus, or aperture");
        vec3 w = to_unit_vector(settings.lookfrom - settings.lookat);
        vec3 u = to_unit_vector(cross(settings.vup, w)), v = cross(w, u);
        double viewport_height =
            2 * std::tan(degrees_to_radians(settings.vfov) / 2) * settings.focus_dist;
        vec3 viewport_u = viewport_height * (double(width) / height) * u,
             viewport_v = -viewport_height * v;
        du_ = viewport_u / width;
        dv_ = viewport_v / height;
        pixel00_ =
            center_ - settings.focus_dist * w - viewport_u / 2 - viewport_v / 2 + 0.5 * (du_ + dv_);
        double radius =
            settings.focus_dist * std::tan(degrees_to_radians(settings.defocus_angle) / 2);
        disk_u_ = radius * u;
        disk_v_ = radius * v;
        defocus_ = radius > 0;
    }
    ray get_ray(int x, int y, sampler &rng) const {
        double dx = rng.uniform() - 0.5, dy = rng.uniform() - 0.5;
        point3 pixel = pixel00_ + (x + dx) * du_ + (y + dy) * dv_, origin = center_;
        if (defocus_) {
            auto p = random_in_unit_disk(rng);
            origin += p.x() * disk_u_ + p.y() * disk_v_;
        }
        return ray(origin, to_unit_vector(pixel - origin));
    }
    ray center_ray(int x, int y) const {
        return ray(center_, to_unit_vector(pixel00_ + x * du_ + y * dv_ - center_));
    }

  private:
    point3 center_, pixel00_;
    vec3 du_, dv_, disk_u_, disk_v_;
    bool defocus_;
};
