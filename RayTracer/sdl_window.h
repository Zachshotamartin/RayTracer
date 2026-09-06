#pragma once
#include "framebuffer.h"
#include "renderer.h"
#include <memory>
#include <string>
struct window_actions {
    bool quit = false, cancel = false, pause = false, restart = false, save = false,
         denoise = false, caustics = false, glass_shadows = false;
    int scene_index = -1;
    double exposure_delta = 0;
};
class sdl_window {
  public:
    sdl_window(int width, int height);
    ~sdl_window();
    sdl_window(const sdl_window &) = delete;
    sdl_window &operator=(const sdl_window &) = delete;
    window_actions poll();
    std::filesystem::path preferred_output_directory() const;
    void present(const frame_snapshot &, const render_stats &, const std::string &scene,
                 const std::string &status, int target_samples, double exposure,
                 const std::string &notice);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
