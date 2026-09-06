#pragma once
#include "scene.h"
scene read_scene_file(const std::filesystem::path &path);
void write_scene_file(const std::filesystem::path &path, const std::string &preset,
                      std::uint64_t scene_seed, const camera_settings &view);
