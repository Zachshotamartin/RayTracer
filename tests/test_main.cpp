#include "headers_and_constants.h"
#include "renderer.h"
#include <chrono>
#include <fstream>
#include <functional>
#include <thread>

namespace {
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
void near(double a, double b, double tolerance = 1e-9) {
    require(std::fabs(a - b) <= tolerance,
            "Expected " + std::to_string(b) + ", got " + std::to_string(a));
}
void near(const vec3 &a, const vec3 &b, double tolerance = 1e-9) {
    for (int i = 0; i < 3; ++i)
        near(a[i], b[i], tolerance);
}
auto white() {
    return std::make_shared<lambertian>(color(1, 1, 1));
}
frame_snapshot render(const std::string &name, int threads, bool bvh = true, int width = 64,
                      int samples = 12) {
    render_settings opts;
    opts.width = width;
    opts.samples = samples;
    opts.max_depth = 12;
    opts.threads = threads;
    opts.use_bvh = bvh;
    render_session session(make_scene(name, opts.seed), opts);
    session.wait();
    auto result = session.capture();
    require(result.frame.samples == samples, "Incomplete render");
    require(result.stats.path_rays >=
                std::uint64_t(result.frame.width) * result.frame.height * samples,
            "Ray counters missing paths");
    return result.frame;
}
void same_image(const frame_snapshot &a, const frame_snapshot &b) {
    require(a.linear.size() == b.linear.size(), "Image sizes differ");
    for (std::size_t i = 0; i < a.linear.size(); ++i)
        near(a.linear[i], b.linear[i], 1e-11);
}
void sphere_hits() {
    sphere s(point3(0, 0, 0), 1, white());
    hit_record rec;
    require(s.hit(ray(point3(3, 0, 0), vec3(-1, 0, 0)), interval(0, infinity), rec),
            "Sphere entry missing");
    near(rec.t, 2);
    near(rec.normal, vec3(1, 0, 0));
    require(rec.front_face, "Entry face flipped");
    require(s.hit(ray(point3(0, 0, 0), vec3(0, 1, 0)), interval(0, infinity), rec),
            "Sphere exit missing");
    near(rec.t, 1);
    near(rec.normal, vec3(0, -1, 0));
    require(!rec.front_face, "Exit face flipped");
    require(!s.hit(ray(point3(3, 0, 0), vec3(0, 1, 0)), interval(0, infinity), rec),
            "False sphere hit");
    require(!s.hit(ray(point3(3, 0, 0), vec3(0, 0, 0)), interval(0, infinity), rec),
            "Zero ray hit sphere");
}
void cube_hits() {
    vec3 u(1, 0, 0), v(0, 1, 0), w(0, 0, 1);
    hit_record rec;
    cube moved(point3(2, 3, 4), vec3(2, 2, 2), u, v, w, white());
    require(moved.hit(ray(point3(5, 3, 4), -u), interval(0.001, infinity), rec),
            "Translated box missed");
    near(rec.t, 2);
    near(rec.normal, u);
    cube centered(point3(0, 0, 0), vec3(2, 2, 2), u, v, w, white());
    require(centered.hit(ray(point3(0, 0, 0), u), interval(0.001, infinity), rec),
            "Box exit missing");
    near(rec.t, 1);
    near(rec.normal, -u);
    require(!rec.front_face, "Box exit marked front");
    require(centered.hit(ray(point3(-3, 1, 0), u), interval(0.001, infinity), rec),
            "Parallel boundary ray missed");
    near(rec.t, 2);
    require(!centered.hit(ray(point3(-3, 1.01, 0), u), interval(0.001, infinity), rec),
            "Outside parallel ray hit");
    require(centered.hit(ray(point3(-3, 0, 0), u), interval(2.5, 10), rec),
            "Clipped entry lost exit");
    near(rec.t, 4);
    double a = 0.63;
    u = vec3(std::cos(a), 0, -std::sin(a));
    w = vec3(std::sin(a), 0, std::cos(a));
    point3 center(2, 3, 4);
    cube rotated(center, vec3(2, 2, 2), u, v, w, white());
    require(rotated.hit(ray(center + 3 * u, -u), interval(0, infinity), rec), "Rotated box missed");
    near(rec.t, 2);
    near(rec.normal, u);
}
void bounds_and_bvh() {
    aabb bounds(point3(-1, -1, -1), point3(1, 1, 1));
    require(bounds.hit(ray(point3(-3, 1, 0), vec3(1, 0, 0)), interval(0, infinity)),
            "AABB boundary miss");
    require(!bounds.hit(ray(point3(-3, 2, 0), vec3(1, 0, 0)), interval(0, infinity)),
            "AABB parallel false positive");
    auto s = make_scene("field");
    bvh_node tree(s.objects.objects);
    sampler rng(19);
    for (int i = 0; i < 3000; ++i) {
        vec3 origin = vec3::random(rng, -15, 15), direction = random_unit_vector(rng);
        ray r(origin, direction);
        hit_record a, b;
        bool linear = s.objects.hit(r, interval(1e-8, infinity), a),
             accelerated = tree.hit(r, interval(1e-8, infinity), b);
        require(linear == accelerated, "BVH disagrees with linear traversal");
        if (linear) {
            near(a.t, b.t);
            near(a.normal, b.normal);
            require(a.mat == b.mat, "BVH material mismatch");
        }
    }
}
void shadows() {
    sampler rng;
    PointLight source(point3(0, 1, 0), color(1, 1, 1), 1);
    auto sample = source.sample(point3(0, 0, 0), rng);
    hittable_list world;
    require(light_list::visible(point3(0, 0, 0), sample, world), "Empty world shadowed");
    world.add(std::make_shared<sphere>(point3(0, 3, 0), 0.5, white()));
    require(light_list::visible(point3(0, 0, 0), sample, world),
            "Object behind point light casts shadow");
    world.add(std::make_shared<sphere>(point3(0, 0.5, 0), 0.1, white()));
    require(!light_list::visible(point3(0, 0, 0), sample, world), "Occluder before light ignored");
    auto distant = source.sample(point3(0, -1, 0), rng);
    near(distant.radiance, color(0.25, 0.25, 0.25));
    auto at_source = source.sample(point3(0, 1, 0), rng);
    require(at_source.pdf == 0, "Point light singularity");
}
void materials() {
    hit_record rec;
    rec.p = point3(0, 0, 0);
    rec.normal = vec3(0, 0, 1);
    sampler rng(99);
    dielectric glass(1.5);
    rec.front_face = false;
    vec3 incoming(0.9, 0, -std::sqrt(0.19));
    for (int i = 0; i < 100; ++i) {
        scatter_sample out;
        require(glass.sample(incoming, rec, rng, out), "TIR sample failed");
        near(out.direction, reflect(incoming, rec.normal));
        near(out.weight, color(1, 1, 1));
    }
    near(dielectric::reflectance(1, 1.5), 0.04);
    lambertian diffuse(color(0.2, 0.4, 0.6));
    double cosine = 0;
    for (int i = 0; i < 20000; ++i) {
        scatter_sample out;
        require(diffuse.sample(vec3(0, 0, -1), rec, rng, out), "Diffuse sample failed");
        require(dot(out.direction, rec.normal) >= 0, "Diffuse sample below surface");
        near(out.weight, color(0.2, 0.4, 0.6));
        near(out.pdf, diffuse.pdf(vec3(0, 0, -1), out.direction, rec));
        cosine += dot(out.direction, rec.normal);
    }
    near(cosine / 20000, 2.0 / 3, 0.01);
    metal rough(color(0.8, 0.6, 0.3), 0.4);
    for (int i = 0; i < 5000; ++i) {
        scatter_sample out;
        if (rough.sample(vec3(0, 0, -1), rec, rng, out)) {
            require(out.pdf > 0 && std::isfinite(out.pdf), "GGX invalid PDF");
            near(out.weight, rough.evaluate(vec3(0, 0, -1), out.direction, rec) *
                                 (out.direction.z() / out.pdf));
        }
    }
}
void additive_light() {
    hittable_list world;
    world.add(std::make_shared<sphere>(point3(0, 0, 0), 1, white()));
    environment env;
    env.sky = false;
    env.background = color(1, 1, 1);
    light_list lights;
    ray r(point3(0, 0, 3), vec3(0, 0, -1));
    ray_counts counts;
    sampler rng(10);
    near(trace_path(r, world, lights, env, 2, rng, counts), color(1, 1, 1));
    lights.add(std::make_shared<DirectionalLight>(vec3(0, 0, -1), color(1, 1, 1), rt_pi));
    sampler rng2(10);
    near(trace_path(r, world, lights, env, 2, rng2, counts), color(2, 2, 2));
    world.clear();
    world.add(std::make_shared<sphere>(point3(0, 0, 0), 1,
                                       std::make_shared<metal>(color(0.8, 0.6, 0.2), 0)));
    light_list none;
    sampler rng3(10);
    near(trace_path(r, world, none, env, 2, rng3, counts), color(0.8, 0.6, 0.2));
}
void area_light_energy() {
    hittable_list world;
    light_list lights;
    environment env;
    env.sky = false;
    world.add(std::make_shared<quad>(point3(-10, 0, 10), vec3(20, 0, 0), vec3(0, 0, -20), white()));
    auto panel = std::make_shared<quad>(point3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2),
                                        std::make_shared<diffuse_light>(color(1, 1, 1)));
    world.add(panel);
    lights.add(std::make_shared<AreaLight>(panel, color(1, 1, 1)));
    double reference = 0;
    constexpr int grid = 200;
    for (int y = 0; y < grid; ++y)
        for (int x = 0; x < grid; ++x) {
            double px = -1 + 2 * (x + 0.5) / grid, pz = -1 + 2 * (y + 0.5) / grid,
                   r2 = px * px + pz * pz + 4;
            reference += 16 / (rt_pi * r2 * r2 * grid * grid);
        }
    for (int depth : {1, 2}) {
        color sum;
        ray_counts counts;
        for (int sample = 0; sample < 30000; ++sample) {
            auto rng = sampler::for_pixel(38, 0, sample);
            sum += trace_path(ray(point3(0, 1, 0), vec3(0, -1, 0)), world, lights, env, depth, rng,
                              counts);
        }
        near(sum / 30000, color(reference, reference, reference), 0.006);
    }
}
void determinism() {
    auto single = render("field", 1), multiple = render("field", 4),
         linear = render("field", 4, false);
    same_image(single, multiple);
    same_image(single, linear);
    same_image(render("studio", 1), render("studio", 4));
}
void cancellation() {
    render_settings settings;
    settings.width = 128;
    settings.samples = 100000;
    settings.threads = 4;
    render_session session(make_scene("field"), settings);
    session.set_paused(true);
    require(session.paused(), "Pause flag missing");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto before = session.stats().samples;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    require(session.stats().samples == before, "Paused renderer kept publishing");
    session.set_paused(false);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (session.stats().samples <= before && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(session.stats().samples > before, "Resume made no progress");
    session.set_paused(true);
    auto start = std::chrono::steady_clock::now();
    session.cancel();
    session.wait();
    require(session.done(), "Cancellation did not finish");
    require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
            "Cancellation was unresponsive");
    auto result = session.capture();
    require(result.frame.samples == result.stats.samples, "Snapshot and statistics disagree");
    // Exercise cancellation at barriers and during thread startup.
    for (int i = 0; i < 12; ++i) {
        render_session short_lived(make_scene("demo"), settings);
        short_lived.cancel();
        short_lived.wait();
    }
}
void validation() {
    for (auto bad : {0, -1, 100001}) {
        render_settings settings;
        settings.samples = bad;
        bool rejected = false;
        try {
            settings.validate(16.0 / 9);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "Bad sample count accepted");
    }
    bool rejected = false;
    try {
        sphere invalid(point3(), 0, white());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "Zero-radius sphere accepted");
    rejected = false;
    try {
        camera_settings settings;
        settings.lookfrom = settings.lookat;
        camera invalid(settings, 64, 36);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "Degenerate camera accepted");
}
void golden_images(bool update) {
    for (const auto &name : scene_names()) {
        auto frame = render(name, 2, true, 64, 32);
        auto pixels = frame.rgb();
        auto path = std::filesystem::path(RT_SOURCE_DIR) / "tests" / "reference" / (name + ".ppm");
        if (update) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            out << "P6\n" << frame.width << " " << frame.height << "\n255\n";
            out.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
            require(bool(out), "Could not write reference");
            continue;
        }
        std::ifstream in(path, std::ios::binary);
        std::string magic;
        int width = 0, height = 0, max = 0;
        in >> magic >> width >> height >> max;
        in.get();
        require(magic == "P6" && width == frame.width && height == frame.height && max == 255,
                "Missing or invalid " + name + " reference");
        std::vector<unsigned char> expected(pixels.size());
        in.read(reinterpret_cast<char *>(expected.data()), expected.size());
        require(bool(in), "Truncated reference");
        double error = 0;
        int large = 0;
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            int difference = std::abs(int(pixels[i]) - int(expected[i]));
            error += difference;
            large += difference > 8;
        }
        require(error / pixels.size() < 0.75 && double(large) / pixels.size() < 0.005,
                name + " image regression exceeded tolerance");
    }
}
} // namespace
int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--update-references") {
        golden_images(true);
        std::cout << "Updated reference images; visually inspect before committing.\n";
        return 0;
    }
    std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"sphere entry, exit, and misses", sphere_hits},
        {"translated, rotated, and inside box hits", cube_hits},
        {"parallel bounds and BVH agreement", bounds_and_bvh},
        {"finite light visibility and falloff", shadows},
        {"dielectric TIR and material sampling", materials},
        {"additive direct and reflected light", additive_light},
        {"area-light MIS energy against quadrature", area_light_energy},
        {"worker and BVH determinism", determinism},
        {"pause, resume, cancellation, and teardown", cancellation},
        {"input validation", validation},
        {"three scene image regressions", [] { golden_images(false); }}};
    int failed = 0;
    for (const auto &[name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception &error) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    return failed ? 1 : 0;
}
