#include "bvh.h"
#include "caustics.h"
#include "cube.h"
#include "denoiser.h"
#include "mesh.h"
#include "renderer.h"
#include "triangle.h"
#include <chrono>
#include <fstream>
#include <functional>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void near(double a, double b, double tolerance = 1e-8) {
    require(std::fabs(a - b) <= tolerance, "Numeric comparison failed");
}
auto white() {
    return std::make_shared<lambertian>(color(1, 1, 1));
}
void geometry() {
    triangle t({point3(0, 0, 0), point3(2, 0, 0), point3(0, 2, 0)}, white(),
               {vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)});
    hit_record rec;
    require(t.hit(ray(point3(0.5, 0.5, 2), vec3(0, 0, -1)), interval(0, infinity), rec),
            "Triangle front missed");
    near(rec.t, 2);
    near(rec.u, 0.25);
    near(rec.v, 0.25);
    require(rec.front_face, "Front face flipped");
    require(t.hit(ray(point3(0.5, 0.5, -2), vec3(0, 0, 1)), interval(0, infinity), rec),
            "Triangle back missed");
    require(!rec.front_face, "Back face flipped");
    near(rec.geometric_normal.z(), -1);
    require(!t.hit(ray(point3(2, 2, 2), vec3(0, 0, -1)), interval(0, infinity), rec),
            "Outside triangle hit");
    require(!t.hit(ray(point3(0.5, 0.5, 2), vec3(1, 0, 0)), interval(0, infinity), rec),
            "Parallel triangle hit");
    auto mesh = load_obj(std::filesystem::path(RT_SOURCE_DIR) / "assets/meshes/pedestal.obj", true);
    require(mesh.triangles == 14, "OBJ polygons not triangulated");
    near(mesh.bounds.min.y(), 0);
    near(mesh.bounds.max.y(), 2);
    bvh_node bvh(mesh.geometry.objects);
    sampler rng(200);
    for (int i = 0; i < 1000; ++i) {
        ray r(vec3::random(rng, -4, 4), random_unit_vector(rng));
        hit_record a, b;
        bool hit = mesh.geometry.hit(r, interval(1e-8, infinity), a);
        require(hit == bvh.hit(r, interval(1e-8, infinity), b), "Mesh BVH disagrees");
        if (hit)
            near(a.t, b.t);
    }
    auto root = std::filesystem::temp_directory_path() /
                ("raytracer-obj-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try {
        {
            std::ofstream out(root / "quad.obj");
            out << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 "
                   "1\nf -4/1/1 -3/2/1 -2/3/1 -1/4/1\n";
        }
        require(load_obj(root / "quad.obj").triangles == 2, "Negative OBJ indices failed");
        {
            std::ofstream out(root / "bad.obj");
            out << "v 0 0 0\nf 1 2 999\n";
        }
        bool rejected = false;
        try {
            load_obj(root / "bad.obj");
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Invalid OBJ accepted");
        std::filesystem::remove_all(root);
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }
}
void filtering() {
    frame_snapshot input{64, 32, 4, std::vector<color>(64 * 32)};
    input.guides.resize(input.linear.size());
    sampler rng(3);
    double before = 0;
    for (int y = 0; y < input.height; ++y)
        for (int x = 0; x < input.width; ++x) {
            auto i = y * input.width + x;
            double value = x < 32 ? 0.3 : 0.8;
            double noise = rng.uniform(-0.2, 0.2);
            input.linear[i] = color(value + noise, value + noise, value + noise);
            input.guides[i] = {vec3(0, 0, 1), color(value, value, value), 2, true, true};
            before += noise * noise;
        }
    auto filtered = denoise(input);
    double after = 0;
    for (int y = 0; y < input.height; ++y)
        for (int x = 0; x < input.width; ++x) {
            auto i = y * input.width + x;
            double value = x < 32 ? 0.3 : 0.8;
            after += std::pow(filtered.linear[i].x() - value, 2);
        }
    require(after < before * 0.35, "Filter failed to reduce noise while preserving the edge");
    require(filtered.denoised && !input.denoised, "Filter mutated the input");
    input.guides[0].filterable = false;
    auto protected_pixel = denoise(input);
    near(protected_pixel.linear[0].x(), input.linear[0].x());
}
void transmission() {
    hittable_list world;
    auto glass = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<cube>(point3(0, 1.5, 0), vec3(2, 1, 2), vec3(1, 0, 0), vec3(0, 1, 0),
                                     vec3(0, 0, 1), glass));
    PointLight point(point3(0, 3, 0), color(1, 1, 1), 1);
    sampler rng;
    auto sample = point.sample(point3(), rng);
    bool crossed = false;
    near(light_list::transmittance(point3(), sample, world, &crossed).x(), 0.96 * 0.96);
    require(crossed, "Glass crossings not recorded");
    require(!light_list::visible(point3(), sample, world),
            "Physical straight connection crossed refractive glass");
    world.add(std::make_shared<quad>(point3(-2, 4, -2), vec3(4, 0, 0), vec3(0, 0, 4), white()));
    near(light_list::transmittance(point3(), sample, world).x(), 0.96 * 0.96);
    world.add(std::make_shared<quad>(point3(-2, 2.5, -2), vec3(4, 0, 0), vec3(0, 0, 4), white()));
    near(light_list::transmittance(point3(), sample, world).x(), 0);
    hit_record rec;
    rec.normal = vec3(0, 1, 0);
    rec.front_face = true;
    for (int i = 0; i < 1000; ++i) {
        scatter_sample out;
        glass->sample_photon(vec3(0, -1, 0), rec, rng, out);
        near(out.weight.x(), 1); // Flux transport must not lose eta^2 at an interface.
    }
}
void photons() {
    hittable_list world;
    light_list lights;
    lights.add(std::make_shared<PointLight>(point3(0, 3, 0), color(1, 1, 1), 10));
    world.add(std::make_shared<quad>(point3(-10, 0, 10), vec3(20, 0, 0), vec3(0, 0, -20), white()));
    caustic_map empty(world, lights, 1000, 12, 0.6, 11);
    require(empty.stored() == 0, "Direct diffuse light incorrectly stored as caustics");
    world.add(std::make_shared<cube>(point3(0, 1.005, 0), vec3(10, 0.01, 10), vec3(1, 0, 0),
                                     vec3(0, 1, 0), vec3(0, 0, 1),
                                     std::make_shared<dielectric>(1.5)));
    caustic_map map(world, lights, 120000, 12, 0.6, 11);
    require(map.stored() > 1000 && map.emitted == 120000,
            "Photon pass failed to transmit through glass");
    hit_record rec;
    require(world.hit(ray(point3(0, 0.5, 0), vec3(0, -1, 0)), interval(0, infinity), rec),
            "Floor hit missing");
    double estimate = map.radiance(vec3(0, -1, 0), rec).x();
    // Thin parallel glass tends to the analytic point-light value times two-interface Fresnel
    // transmission.
    double analytic = 10 / (9 * rt_pi) * 0.96 * 0.96;
    near(estimate, analytic, analytic * 0.22);
    near(map.radiance(vec3(0, -1, 0), rec, 1).x(), 0);
    caustic_map more(world, lights, 240000, 12, 0.6, 11);
    near(more.radiance(vec3(0, -1, 0), rec).x(), estimate, analytic * 0.2);
    // Cancelling the prepass releases every worker and publishes no partial pass.
    render_settings settings;
    settings.width = 32;
    settings.samples = 100;
    settings.threads = 4;
    settings.caustic_photons = 10000000;
    render_session session(make_scene("caustics"), settings);
    session.cancel();
    session.wait();
    require(session.done(), "Photon prepass cancellation failed");
    settings.caustic_photons = 20000;
    settings.samples = 2;
    settings.collect_guides = true;
    render_session first(make_scene("caustics"), settings);
    first.wait();
    settings.threads = 1;
    render_session second(make_scene("caustics"), settings);
    second.wait();
    auto a = first.snapshot(), b = second.snapshot();
    require(a.rgb() == b.rgb(), "Photon rendering depends on worker count");
    require(a.guides.size() == a.linear.size(), "Guide buffer missing");
}
void caustic_partition() {
    hittable_list world;
    light_list lights;
    environment env;
    env.sky = false;
    world.add(std::make_shared<quad>(point3(-5, 0, 5), vec3(10, 0, 0), vec3(0, 0, -10), white()));
    world.add(std::make_shared<cube>(point3(0, 1.005, 0), vec3(6, 0.01, 6), vec3(1, 0, 0),
                                     vec3(0, 1, 0), vec3(0, 0, 1),
                                     std::make_shared<dielectric>(1.5)));
    auto panel = std::make_shared<quad>(point3(-1, 3, -1), vec3(2, 0, 0), vec3(0, 0, 2),
                                        std::make_shared<diffuse_light>(color(1, 1, 1)));
    world.add(panel);
    lights.add(std::make_shared<AreaLight>(panel, color(1, 1, 1)));
    caustic_map map(world, lights, 150000, 12, 0.3, 7);
    color baseline, mapped;
    ray view(point3(0, 0.5, 0), vec3(0, -1, 0));
    for (int i = 0; i < 30000; ++i) {
        sampler a(i + 50), b(i + 50);
        ray_counts counts;
        baseline += trace_path(view, world, lights, env, 4, a, counts);
        mapped += trace_path(view, world, lights, env, 4, b, counts, {false, &map});
    }
    baseline /= 30000;
    mapped /= 30000;
    require(baseline.x() > 0.04, "Independent path-traced caustic reference is empty");
    near(mapped.x(), baseline.x(), baseline.x() * 0.16);
}
void fixtures(const std::filesystem::path &root) {
    frame_snapshot frame{16, 2, 1, std::vector<color>(32, color(0.5, 2, 8))};
    frame.linear[0] = color(0, 0, 0);
    frame.linear[16] = color(100, 25, 1);
    write_png(root / "fixture.png", frame);
    write_hdr(root / "fixture.hdr", frame);
    write_pfm(root / "fixture.pfm", frame);
    frame_snapshot flat{128, 128, 1, std::vector<color>(128 * 128, color(0.5, 0.5, 0.5))};
    write_png(root / "compressed.png", flat);
    require(std::filesystem::file_size(root / "compressed.png") < 4096,
            "PNG did not compress flat image");
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-fixtures") {
            fixtures(argv[2]);
            return 0;
        }
        geometry();
        std::cout << "PASS triangles, OBJ/MTL, indices, and mesh BVH\n";
        filtering();
        std::cout << "PASS denoising noise reduction and edge preservation\n";
        transmission();
        std::cout << "PASS finite glass transmittance and photon transport\n";
        photons();
        std::cout << "PASS caustic energy, normalization, determinism, and cancellation\n";
        caustic_partition();
        std::cout << "PASS photon/path-tracer energy partition against independent transport\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
