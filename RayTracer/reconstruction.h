#pragma once
#include "renderer.h"
#include <memory>
#include <optional>
#include <string>

void append_reprojected_history(const frame_snapshot &current, const frame_snapshot &previous,
                                const std::vector<color> &previous_rgb, std::vector<float> &input);

class neural_denoiser {
  public:
    explicit neural_denoiser(const std::filesystem::path &model,
                             const std::string &provider = "cpu");
    ~neural_denoiser();
    frame_snapshot reconstruct(const frame_snapshot &frame);
    int scale() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
struct reconstruction_result {
    frame_snapshot frame, raw;
    render_stats stats;
    std::uint64_t generation = 0;
    double seconds = 0;
    std::string error;
};
// One worker owns the model. New requests replace the pending request, never raw samples.
class reconstruction_worker {
  public:
    reconstruction_worker(const std::filesystem::path &model, const std::string &provider);
    ~reconstruction_worker();
    void submit(const frame_snapshot &frame, const render_stats &stats, std::uint64_t generation);
    std::optional<reconstruction_result> poll();

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
