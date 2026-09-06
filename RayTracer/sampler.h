#pragma once
#include <cstdint>
// Explicit per-path SplitMix64 streams; no shared state or library distributions.
class sampler {
  public:
    explicit sampler(std::uint64_t seed = 42) : state_(seed) {}
    static std::uint64_t mix(std::uint64_t x) {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }
    double uniform() {
        state_ += UINT64_C(0x9e3779b97f4a7c15);
        return static_cast<double>(mix(state_) >> 11) * 0x1.0p-53;
    }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
    static sampler for_pixel(std::uint64_t seed, std::uint64_t pixel, std::uint64_t sample) {
        return sampler(mix(seed) ^ mix(pixel + UINT64_C(0x632be59bd9b4e019)) ^
                       mix(sample + UINT64_C(0x8cb92baa3f3d8dd7)));
    }

  private:
    std::uint64_t state_;
};
