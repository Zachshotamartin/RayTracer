#pragma once
#include "hittable_list.h"
#include <filesystem>
struct mesh_data {
    hittable_list geometry;
    aabb bounds;
    std::size_t triangles = 0;
    std::string warnings;
};
// Normalize the longest axis to two units and place the mesh on y=0 when requested.
mesh_data load_obj(const std::filesystem::path &path, bool normalize = false);
