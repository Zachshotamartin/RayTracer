#pragma once
#include "camera.h"
#include "hittable_list.h"
#include "integrator.h"
#include <filesystem>
#include <string>
#include <vector>
struct scene {
    std::string name;
    hittable_list objects;
    light_list lights;
    camera_settings view;
    environment env;
};
scene make_scene(const std::string &name, std::uint64_t seed = 42);
scene make_mesh_scene(const std::filesystem::path &path, std::string &warnings);
const std::vector<std::string> &scene_names();
