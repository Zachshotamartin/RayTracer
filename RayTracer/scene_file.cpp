#include "scene_file.h"
#include "../third_party/json.hpp"
#include "cube.h"
#include "mesh.h"
#include "sphere.h"
#include "triangle.h"
#include <fstream>
#include <numbers>

namespace {
using json = nlohmann::json;
class annotated_surface : public hittable {
    std::shared_ptr<hittable> surface_;

  public:
    explicit annotated_surface(std::shared_ptr<hittable> surface) : surface_(std::move(surface)) {}
    bool hit(const ray &r, interval range, hit_record &rec) const override {
        if (!surface_->hit(r, range, rec))
            return false;
        rec.annotation = 1;
        return true;
    }
    aabb bounding_box() const override { return surface_->bounding_box(); }
};
vec3 vector3(const json &j) {
    if (!j.is_array() || j.size() != 3)
        throw std::invalid_argument("Expected a three-component vector");
    vec3 v(j.at(0).get<double>(), j.at(1).get<double>(), j.at(2).get<double>());
    for (int c = 0; c < 3; ++c)
        if (!std::isfinite(v[c]) || std::abs(v[c]) > 1e6)
            throw std::invalid_argument("Scene vector must be finite and bounded");
    return v;
}
color rgb(const json &j, double maximum = 1) {
    auto c = vector3(j);
    if (std::min({c.x(), c.y(), c.z()}) < 0 || std::max({c.x(), c.y(), c.z()}) > maximum)
        throw std::invalid_argument("Scene color outside its supported range");
    return c;
}
double number(const json &j, const char *key, double fallback, double low, double high) {
    double value = j.value(key, fallback);
    if (!std::isfinite(value) || value < low || value > high)
        throw std::invalid_argument(std::string("Invalid scene field: ") + key);
    return value;
}
json array(const vec3 &v) {
    return json::array({v.x(), v.y(), v.z()});
}
std::shared_ptr<material> surface(const json &j) {
    auto type = j.value("type", "diffuse");
    if (type == "glass")
        return std::make_shared<dielectric>(number(j, "ior", 1.5, 1.01, 3));
    auto albedo = rgb(j.at("albedo"));
    if (type == "diffuse" && j.contains("texture")) {
        const auto &t = j.at("texture");
        const auto kind = t.at("type").get<std::string>();
        const auto other = rgb(t.at("color"));
        if (kind == "checker")
            return std::make_shared<lambertian>(std::make_shared<checker_texture>(
                number(t, "scale", .25, .0001, 1e4), albedo, other));
        if (kind == "bands")
            return std::make_shared<lambertian>(std::make_shared<band_texture>(
                albedo, other, number(t, "frequency", 12, .01, 1e4)));
        throw std::invalid_argument("Unknown scene texture: " + kind);
    }
    if (type == "diffuse")
        return std::make_shared<lambertian>(albedo);
    if (type == "metal")
        return std::make_shared<metal>(albedo, number(j, "roughness", .2, 0, 1));
    throw std::invalid_argument("Unknown scene material: " + type);
}
void camera_from(const json &j, camera_settings &v) {
    if (j.contains("lookfrom"))
        v.lookfrom = vector3(j.at("lookfrom"));
    if (j.contains("lookat"))
        v.lookat = vector3(j.at("lookat"));
    if (j.contains("up"))
        v.vup = vector3(j.at("up"));
    v.aspect_ratio = number(j, "aspect_ratio", v.aspect_ratio, .1, 10);
    v.vfov = number(j, "vfov", v.vfov, 1, 170);
    v.focus_dist = number(j, "focus_dist", v.focus_dist, .001, 1e6);
    v.defocus_angle = number(j, "defocus_angle", v.defocus_angle, 0, 30);
    if ((v.lookfrom - v.lookat).length() < 1e-6 ||
        cross(v.lookfrom - v.lookat, v.vup).length() < 1e-6)
        throw std::invalid_argument("Degenerate scene camera");
}
} // namespace
scene read_scene_file(const std::filesystem::path &path) {
    if (std::filesystem::file_size(path) > 4 * 1024 * 1024)
        throw std::invalid_argument("Scene configuration exceeds 4 MiB");
    std::ifstream input(path);
    auto j = json::parse(input);
    if (j.at("schema_version") != 1)
        throw std::invalid_argument("Unsupported scene schema");
    scene s;
    if (j.contains("preset"))
        s = make_scene(j.at("preset"), j.value("scene_seed", std::uint64_t(42)));
    else {
        s.name = j.value("name", "custom");
        s.env.sky = false;
        if (!j.at("objects").is_array() || j.at("objects").size() > 10000)
            throw std::invalid_argument("Invalid scene object count");
        for (const auto &o : j.at("objects")) {
            const auto first_object = s.objects.objects.size();
            auto mat = surface(o.at("material"));
            std::string type = o.at("type");
            if (type == "sphere")
                s.objects.add(std::make_shared<sphere>(vector3(o.at("center")),
                                                       number(o, "radius", 1, .0001, 1e5), mat));
            else if (type == "quad")
                s.objects.add(std::make_shared<quad>(vector3(o.at("origin")), vector3(o.at("u")),
                                                     vector3(o.at("v")), mat));
            else if (type == "box") {
                double angle =
                    number(o, "rotation_y", 0, -2 * std::numbers::pi, 2 * std::numbers::pi);
                auto size = vector3(o.at("size"));
                if (std::min({size.x(), size.y(), size.z()}) <= 0)
                    throw std::invalid_argument("Box size must be positive");
                s.objects.add(std::make_shared<cube>(
                    vector3(o.at("center")), size, vec3(std::cos(angle), 0, -std::sin(angle)),
                    vec3(0, 1, 0), vec3(std::sin(angle), 0, std::cos(angle)), mat));
            } else if (type == "mesh" && o.contains("path")) {
                auto mesh_path = path.parent_path() / o.at("path").get<std::string>();
                auto mesh = load_obj(mesh_path, o.value("normalize", false));
                s.objects.add(std::make_shared<hittable_list>(mesh.geometry));
            } else if (type == "mesh") {
                const auto &vertices = o.at("vertices"), &faces = o.at("faces");
                if (!vertices.is_array() || !faces.is_array() || vertices.size() > 100000 ||
                    faces.empty() || faces.size() > 100000)
                    throw std::invalid_argument("Invalid inline mesh size");
                std::vector<point3> points;
                for (const auto &vertex : vertices)
                    points.push_back(vector3(vertex));
                for (const auto &face : faces) {
                    if (!face.is_array() || face.size() != 3)
                        throw std::invalid_argument("Inline mesh requires triangle faces");
                    std::array<point3, 3> triangle_points;
                    for (int k = 0; k < 3; ++k) {
                        if (!face[k].is_number_unsigned() && !face[k].is_number_integer())
                            throw std::invalid_argument("Mesh index must be an integer");
                        auto index = face[k].get<std::int64_t>();
                        if (index < 0 || std::uint64_t(index) >= points.size())
                            throw std::invalid_argument("Mesh index outside vertex array");
                        triangle_points[k] = points[std::size_t(index)];
                    }
                    if (cross(triangle_points[1] - triangle_points[0],
                              triangle_points[2] - triangle_points[0])
                            .length() < 1e-12)
                        throw std::invalid_argument("Degenerate inline mesh triangle");
                    s.objects.add(std::make_shared<triangle>(triangle_points, mat));
                }
            } else
                throw std::invalid_argument("Unsupported scene object: " + type);
            if (o.value("annotation", "solid") == "thin")
                for (auto index = first_object; index < s.objects.objects.size(); ++index)
                    s.objects.objects[index] =
                        std::make_shared<annotated_surface>(s.objects.objects[index]);
        }
        if (!j.at("lights").is_array() || j.at("lights").empty() || j.at("lights").size() > 128)
            throw std::invalid_argument("Scene needs 1..128 lights");
        for (const auto &l : j.at("lights")) {
            std::string type = l.at("type");
            auto energy = rgb(l.at("emission"), 1e4);
            if (type == "area") {
                auto q = std::make_shared<quad>(vector3(l.at("origin")), vector3(l.at("u")),
                                                vector3(l.at("v")),
                                                std::make_shared<diffuse_light>(energy));
                s.objects.add(q);
                s.lights.add(std::make_shared<AreaLight>(q, energy));
            } else if (type == "point")
                s.lights.add(std::make_shared<PointLight>(vector3(l.at("position")), energy, 1));
            else
                throw std::invalid_argument("Unsupported scene light: " + type);
        }
    }
    if (j.contains("environment")) {
        auto e = j.at("environment");
        s.env.sky = e.value("sky", false);
        s.env.background = rgb(e.value("background", json::array({0., 0., 0.})), 1e4);
    }
    if (j.contains("camera"))
        camera_from(j.at("camera"), s.view);
    return s;
}
void write_scene_file(const std::filesystem::path &path, const std::string &preset,
                      std::uint64_t seed, const camera_settings &v) {
    json j = {{"schema_version", 1},
              {"preset", preset},
              {"scene_seed", seed},
              {"camera",
               {{"lookfrom", array(v.lookfrom)},
                {"lookat", array(v.lookat)},
                {"up", array(v.vup)},
                {"vfov", v.vfov},
                {"aspect_ratio", v.aspect_ratio},
                {"focus_dist", v.focus_dist},
                {"defocus_angle", v.defocus_angle}}}};
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << j.dump(2) << '\n';
    if (!out)
        throw std::runtime_error("Could not write scene configuration");
}
