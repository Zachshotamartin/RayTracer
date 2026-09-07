#pragma once
#include "renderer.h"
#include <memory>
#include <optional>
#include <string>

void append_reprojected_history(const frame_snapshot &current, const frame_snapshot &previous,
                                const std::vector<color> &previous_rgb, std::vector<float> &input,
                                int schema = 1);

struct neural_settings {
    int threads = 2;
    std::filesystem::path cache_directory, profile_prefix;
};
struct neural_stats {
    double packing_seconds = 0, inference_seconds = 0, output_seconds = 0;
    std::size_t input_bytes = 0;
};

class neural_denoiser {
  public:
    explicit neural_denoiser(const std::filesystem::path &model,
                             const std::string &provider = "cpu", neural_settings settings = {});
    ~neural_denoiser();
    frame_snapshot reconstruct(const frame_snapshot &frame);
    int scale() const;
    neural_stats stats() const;

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
    neural_stats details;
};
// One worker owns the model. New requests replace the pending request, never raw samples.
class reconstruction_worker {
  public:
    reconstruction_worker(const std::filesystem::path &model, const std::string &provider,
                          neural_settings settings = {});
    ~reconstruction_worker();
    void submit(const frame_snapshot &frame, const render_stats &stats, std::uint64_t generation);
    std::optional<reconstruction_result> poll();

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
