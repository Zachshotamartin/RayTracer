#ifndef LIGHT_H
#define LIGHT_H

#include "vec3.h"
#include <vector>
#include <memory>
#include "hittable_list.h"
#include "color.h"
using std::shared_ptr;
using std::make_shared;

class light {
public:
    vec3 color;
    float intensity;

    // Constructor
    light(const vec3& color, float intensity)
        : color(color), intensity(intensity) {}

    // Virtual method to get the light direction from a point
    virtual vec3 getDirection(const vec3& point) const = 0;

    // Virtual method to get the light's intensity factor at a point
    virtual float getIntensityAt(const vec3& point) const = 0;

    // Virtual destructor
    virtual ~light() = default;
};

class PointLight : public light {
public:
    vec3 position;

    PointLight(const vec3& position, const vec3& color, float intensity)
        : light(color, intensity), position(position) {}

    // Override method to get light direction from a point
    vec3 getDirection(const vec3& point) const override {
        return to_unit_vector(position - point);
    }

    // Override method to get the intensity (with optional falloff)
    float getIntensityAt(const vec3& point) const override {
        vec3 diff = position - point;
        float distance = std::sqrt(diff.dot());  // Corrected the dot product
        return intensity / (distance * distance);  // Optional falloff
    }
};

class DirectionalLight : public light {
public:
    vec3 direction;

    DirectionalLight(const vec3& direction, const vec3& color, float intensity)
        : light(color, intensity), direction(to_unit_vector(direction)) {}

    // Override method to get light direction (same everywhere)
    vec3 getDirection(const vec3& point) const override {
        return -direction;
    }

    // Intensity is constant for directional light
    float getIntensityAt(const vec3& point) const override {
        return intensity;
    }
};

// Light list (not inheriting from light)
class light_list {
public:
    std::vector<shared_ptr<light>> lights;

    // Constructors
    light_list() = default;
    light_list(shared_ptr<light> light) { add(light); }

    // Add a light to the list
    void add(shared_ptr<light> light) {
        lights.push_back(light);
    }

    // Clear the list of lights
    void clear() {
        lights.clear();
    }

    // Calculate lighting contribution at a point
    color calculate_lighting(const point3& point, const vec3& normal, const hittable& world) {
        vec3 total_light(0.0f, 0.0f, 0.0f);

        for (const auto& light : lights) {
            vec3 light_dir = light->getDirection(point);
            if (!is_in_shadow(point, light_dir, world)) {
                
                // Calculate the contribution (simplified Lambertian shading)
                float diffuse = dot(normal, light_dir);
                float diffuse_factor = diffuse > 0.0f ? diffuse : 0.0f;
                
                // Get the light's intensity at the point
                float intensity = light->getIntensityAt(point);
                
                // Add the diffuse contribution to the total light
                total_light += diffuse_factor * intensity * light->color;
            }
        }

        return total_light;
    }
    
    bool is_in_shadow(const point3& point, const vec3& light_dir, const hittable& world) {
        ray shadow_ray = ray(point, light_dir);  // Ray from the point to the light
        
        hit_record temp_rec;
        if (world.hit(shadow_ray, interval(0.001, infinity), temp_rec)) {
            return true;  // If the shadow ray hits something, the point is in shadow
        }
        
        return false;
    }
};

#endif /* LIGHT_H */
