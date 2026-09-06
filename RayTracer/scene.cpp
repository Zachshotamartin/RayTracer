#include "scene.h"
#include "cube.h"
#include "mesh.h"
#include "sphere.h"

using std::make_shared;
namespace {
void area(scene &s, point3 p, vec3 u, vec3 v, color emission) {
    auto shape = make_shared<quad>(p, u, v, make_shared<diffuse_light>(emission));
    s.objects.add(shape);
    s.lights.add(make_shared<AreaLight>(shape, emission));
}
scene demo() {
    scene s;
    s.name = "demo";
    s.objects.add(make_shared<sphere>(point3(0, -100.5, -1), 100,
                                      make_shared<lambertian>(color(0.8, 0.8, 0))));
    s.objects.add(make_shared<sphere>(point3(0, 0, -1.2), 0.5,
                                      make_shared<lambertian>(color(0.1, 0.2, 0.5))));
    s.objects.add(make_shared<sphere>(point3(-1, 0, -1), 0.5, make_shared<dielectric>(1.5)));
    s.objects.add(make_shared<sphere>(point3(-1, 0, -1), 0.4, make_shared<dielectric>(1 / 1.5)));
    s.objects.add(
        make_shared<sphere>(point3(1, 0, -1), 0.5, make_shared<metal>(color(0.8, 0.6, 0.2), 0)));
    s.lights.add(make_shared<DirectionalLight>(vec3(0, -1, 0), color(1, 0.8, 0.6), 1.5));
    s.view.lookfrom = point3(0, 0, 0);
    s.view.lookat = point3(0, 0, -1);
    s.view.vfov = 90;
    s.view.defocus_angle = 0.6;
    s.view.focus_dist = 1;
    return s;
}
scene field(std::uint64_t seed) {
    // Scene distribution and camera restored from commit 488bb80.
    // The explicit sampler replaces the original hidden global RNG.
    scene s;
    s.name = "field";
    sampler rng(seed);
    s.objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                      make_shared<lambertian>(color(0.5, 0.5, 0.5))));
    for (int a = -11; a < 11; ++a)
        for (int b = -11; b < 11; ++b) {
            double choice = rng.uniform(), x = a + 0.9 * rng.uniform(), z = b + 0.9 * rng.uniform();
            point3 center(x, 0.2, z);
            if ((center - point3(4, 0.2, 0)).length() <= 0.9)
                continue;
            std::shared_ptr<material> mat;
            if (choice < 0.8) {
                color a_color = vec3::random(rng), b_color = vec3::random(rng);
                mat = make_shared<lambertian>(a_color * b_color);
            } else if (choice < 0.95) {
                color albedo = vec3::random(rng, 0.5, 1);
                double roughness = rng.uniform(0, 0.5);
                mat = make_shared<metal>(albedo, roughness);
            } else
                mat = make_shared<dielectric>(1.5);
            s.objects.add(make_shared<sphere>(center, 0.2, mat));
        }
    s.objects.add(make_shared<sphere>(point3(0, 1, 0), 1, make_shared<dielectric>(1.5)));
    s.objects.add(
        make_shared<sphere>(point3(-4, 1, 0), 1, make_shared<lambertian>(color(0.4, 0.2, 0.1))));
    s.objects.add(
        make_shared<sphere>(point3(4, 1, 0), 1, make_shared<metal>(color(0.7, 0.6, 0.5), 0)));
    s.view.lookfrom = point3(13, 2, 3);
    s.view.lookat = point3(0, 0, 0);
    s.view.vfov = 20;
    s.view.defocus_angle = 0.6;
    s.view.focus_dist = 10;
    return s;
}
scene studio() {
    scene s;
    s.name = "studio";
    s.env.sky = false;
    s.env.background = color(0.025, 0.035, 0.045);
    auto floor = make_shared<lambertian>(
        make_shared<checker_texture>(1.3, color(0.36, 0.39, 0.36), color(0.26, 0.29, 0.27)));
    s.objects.add(make_shared<quad>(point3(-10, 0, 8), vec3(20, 0, 0), vec3(0, 0, -20), floor));
    s.objects.add(make_shared<quad>(point3(-10, 0, -3), vec3(20, 0, 0), vec3(0, 8, 0),
                                    make_shared<lambertian>(color(0.035, 0.12, 0.13))));
    auto bands = make_shared<band_texture>(color(0.78, 0.65, 0.38), color(0.11, 0.25, 0.22), 9);
    s.objects.add(
        make_shared<sphere>(point3(-1.55, 1.05, 0), 1.05, make_shared<lambertian>(bands)));
    s.objects.add(make_shared<sphere>(point3(0.8, 1, 0.4), 1, make_shared<dielectric>(1.5)));
    double angle = 0.48;
    s.objects.add(make_shared<cube>(point3(2.7, 0.7, -0.3), vec3(1.3, 1.4, 1.3),
                                    vec3(std::cos(angle), 0, -std::sin(angle)), vec3(0, 1, 0),
                                    vec3(std::sin(angle), 0, std::cos(angle)),
                                    make_shared<metal>(color(0.82, 0.58, 0.26), 0.23)));
    s.objects.add(make_shared<sphere>(point3(-0.5, 0.32, 2), 0.32,
                                      make_shared<metal>(color(0.92, 0.92, 0.92), 0)));
    area(s, point3(-3, 6, -2), vec3(5, 0, 0), vec3(0, 0, 5), color(9, 7.8, 6.2));
    area(s, point3(-5, 5, 1), vec3(0, 0, -3), vec3(0, 2, 0), color(5, 7, 9));
    s.view.lookfrom = point3(5.7, 3, 11);
    s.view.lookat = point3(0.3, 0.9, 0);
    s.view.vfov = 30;
    s.view.focus_dist = (s.view.lookfrom - s.view.lookat).length();
    s.view.defocus_angle = 0.12;
    return s;
}
scene caustics_scene() {
    scene s;
    s.name = "caustics";
    s.env.sky = false;
    auto floor = make_shared<lambertian>(color(0.7, 0.72, 0.75));
    s.objects.add(make_shared<quad>(point3(-10, 0, 10), vec3(20, 0, 0), vec3(0, 0, -20), floor));
    s.objects.add(make_shared<sphere>(point3(0, 1.05, 0), 1, make_shared<dielectric>(1.5)));
    s.objects.add(make_shared<cube>(point3(-2, 0.65, -0.7), vec3(1, 1.3, 1), vec3(1, 0, 0),
                                    vec3(0, 1, 0), vec3(0, 0, 1),
                                    make_shared<lambertian>(color(0.15, 0.3, 0.5))));
    s.lights.add(make_shared<PointLight>(point3(-1.5, 4.5, -2), color(1, 0.93, 0.78), 30));
    s.view.lookfrom = point3(4, 4.5, 7);
    s.view.lookat = point3(0, 0.3, 0);
    s.view.vfov = 40;
    return s;
}
} // namespace
scene make_mesh_scene(const std::filesystem::path &path, std::string &warnings) {
    auto mesh = load_obj(path, true);
    warnings = mesh.warnings;
    scene s;
    s.name = "mesh";
    s.env.sky = false;
    s.env.background = color(0.04, 0.05, 0.06);
    s.objects = std::move(mesh.geometry);
    s.objects.add(make_shared<quad>(point3(-5, -0.01, 5), vec3(10, 0, 0), vec3(0, 0, -10),
                                    make_shared<lambertian>(make_shared<checker_texture>(
                                        0.75, color(0.25, 0.27, 0.3), color(0.45, 0.47, 0.5)))));
    area(s, point3(-3, 5, -2), vec3(5, 0, 0), vec3(0, 0, 5), color(7, 6.5, 5.5));
    s.view.lookfrom = point3(4, 2.8, 5);
    s.view.lookat = point3(0, 0.8, 0);
    s.view.vfov = 35;
    return s;
}
scene make_scene(const std::string &name, std::uint64_t seed) {
    if (name == "demo")
        return demo();
    if (name == "field")
        return field(seed);
    if (name == "studio")
        return studio();
    if (name == "caustics")
        return caustics_scene();
    throw std::invalid_argument("Unknown scene '" + name +
                                "'; choose demo, field, studio, or caustics");
}
const std::vector<std::string> &scene_names() {
    static const std::vector<std::string> names{"demo", "field", "studio", "caustics"};
    return names;
}
