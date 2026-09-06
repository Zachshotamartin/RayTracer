#pragma once
#include "framebuffer.h"
// Spatial, edge-aware a-trous filter. It never modifies the accumulated samples.
frame_snapshot denoise(const frame_snapshot &input, int iterations = 3);
