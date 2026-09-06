#include "mesh.h"
#include "material.h"
#include "triangle.h"
#include <fstream>
#include <sstream>
#define TINYOBJLOADER_IMPLEMENTATION
// Use upstream's portable parser: the embedded fast_float fails MSVC C++20
// constexpr validation. Keep the parser consistent across platforms.
#define TINYOBJLOADER_DISABLE_FAST_FLOAT
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../third_party/tiny_obj_loader.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

mesh_data load_obj(const std::filesystem::path &path, bool normalize) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Could not open OBJ: " + path.string());
    // Stream API preserves native filesystem paths, including Unicode on Windows.
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning, error;
    tinyobj::MaterialFileReader material_reader(path.parent_path().string());
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, &file, &material_reader,
                          true))
        throw std::runtime_error("Invalid OBJ: " + error);
    if (attrib.vertices.empty())
        throw std::invalid_argument("OBJ contains no vertices");
    mesh_data result;
    result.warnings = warning;
    for (std::size_t i = 0; i < attrib.vertices.size(); i += 3) {
        point3 p(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(p[c]))
                throw std::invalid_argument("OBJ has non-finite vertices");
        result.bounds = aabb(result.bounds, aabb(p, p));
    }
    double scale = 1;
    point3 origin;
    if (normalize) {
        vec3 size = result.bounds.max - result.bounds.min;
        double longest = std::max({size.x(), size.y(), size.z()});
        if (!(longest > 0))
            throw std::invalid_argument("OBJ has zero extent");
        scale = 2 / longest;
        origin = (result.bounds.min + result.bounds.max) / 2;
        origin[1] = result.bounds.min.y();
    }
    auto fallback = std::make_shared<lambertian>(color(0.55, 0.65, 0.72));
    std::vector<std::shared_ptr<material>> converted;
    for (const auto &m : materials) {
        if (!std::isfinite(m.ior) || !std::isfinite(m.dissolve) || !std::isfinite(m.roughness) ||
            !std::isfinite(m.metallic))
            throw std::invalid_argument("Non-finite MTL material");
        for (auto c : m.diffuse)
            if (!std::isfinite(c))
                throw std::invalid_argument("Non-finite MTL color");
        color kd(std::clamp(double(m.diffuse[0]), 0.0, 1.0),
                 std::clamp(double(m.diffuse[1]), 0.0, 1.0),
                 std::clamp(double(m.diffuse[2]), 0.0, 1.0));
        if (m.dissolve < 0.99 || m.illum == 4 || m.illum == 6 || m.illum == 7)
            converted.push_back(std::make_shared<dielectric>(m.ior > 1 ? m.ior : 1.5));
        else if (m.metallic > 0.5 || m.illum == 3 || m.illum == 5)
            converted.push_back(std::make_shared<metal>(kd, m.roughness > 0 ? m.roughness : 0.2));
        else
            converted.push_back(std::make_shared<lambertian>(kd));
        if (!m.diffuse_texname.empty())
            result.warnings += "Image textures are not loaded: " + m.diffuse_texname + "\n";
        if (m.emission[0] || m.emission[1] || m.emission[2])
            result.warnings +=
                "MTL emission is not registered as a light; using the surface material.\n";
    }
    aabb transformed;
    for (const auto &shape : shapes) {
        std::size_t offset = 0;
        for (std::size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            auto count = shape.mesh.num_face_vertices[face];
            if (count != 3)
                throw std::invalid_argument("OBJ triangulation failed");
            std::array<point3, 3> points;
            std::array<vec3, 3> uv{}, normals{};
            for (int corner = 0; corner < 3; ++corner) {
                auto idx = shape.mesh.indices.at(offset + corner);
                if (idx.vertex_index < 0 ||
                    std::size_t(idx.vertex_index) >= attrib.vertices.size() / 3)
                    throw std::invalid_argument("OBJ vertex index out of bounds");
                std::size_t v = std::size_t(idx.vertex_index) * 3;
                points[corner] =
                    (point3(attrib.vertices[v], attrib.vertices[v + 1], attrib.vertices[v + 2]) -
                     origin) *
                    scale;
                transformed = aabb(transformed, aabb(points[corner], points[corner]));
                if (idx.texcoord_index >= 0) {
                    std::size_t t = std::size_t(idx.texcoord_index) * 2;
                    if (t + 1 >= attrib.texcoords.size())
                        throw std::invalid_argument("OBJ UV index out of bounds");
                    uv[corner] = vec3(attrib.texcoords[t], attrib.texcoords[t + 1], 0);
                }
                if (idx.normal_index >= 0) {
                    std::size_t n = std::size_t(idx.normal_index) * 3;
                    if (n + 2 >= attrib.normals.size())
                        throw std::invalid_argument("OBJ normal index out of bounds");
                    normals[corner] = to_unit_vector(
                        vec3(attrib.normals[n], attrib.normals[n + 1], attrib.normals[n + 2]));
                }
            }
            auto id = shape.mesh.material_ids.at(face);
            std::shared_ptr<material> mat = id >= 0 ? converted.at(id) : fallback;
            if (cross(points[1] - points[0], points[2] - points[0]).dot() == 0) {
                result.warnings += "Skipped degenerate triangle.\n";
            } else {
                result.geometry.add(std::make_shared<triangle>(points, mat, uv, normals));
                ++result.triangles;
            }
            offset += count;
        }
    }
    if (!result.triangles)
        throw std::invalid_argument("OBJ contains no usable triangles");
    result.bounds = transformed;
    return result;
}
