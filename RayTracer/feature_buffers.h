#pragma once
#include "framebuffer.h"
#include <filesystem>
// Schema 1, top-to-bottom pixels; individual PFM files use standard PFM row order.
void write_feature_buffers(const std::filesystem::path &directory, const frame_snapshot &frame);
std::vector<float> pack_reconstruction_input(const frame_snapshot &frame, int schema = 1);
void pack_reconstruction_input(const frame_snapshot &frame, std::vector<float> &output,
                               int schema = 1);
