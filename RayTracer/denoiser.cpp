#include "denoiser.h"

namespace {
double luminance(const color &c) {
    return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
}
} // namespace
frame_snapshot denoise(const frame_snapshot &input, int iterations) {
    if (input.width < 1 || input.height < 1 ||
        input.linear.size() != std::size_t(input.width) * input.height ||
        input.guides.size() != input.linear.size() || iterations < 1 || iterations > 5)
        throw std::invalid_argument(
            "Denoising requires a matching geometry guide buffer and 1..5 iterations");
    frame_snapshot output = input;
    output.denoised = true;
    std::vector<color> next(input.linear.size());
    // B3-spline wavelet kernel, expanded with holes on successive passes.
    constexpr double kernel[5] = {1, 4, 6, 4, 1};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int step = 1 << iteration;
        for (int y = 0; y < input.height; ++y)
            for (int x = 0; x < input.width; ++x) {
                auto i = std::size_t(y) * input.width + x;
                const auto &guide = input.guides[i];
                // Specular silhouettes and emitters lack reliable first-hit color guides.
                if (!guide.hit || !guide.filterable) {
                    next[i] = output.linear[i];
                    continue;
                }
                double center = luminance(output.linear[i]);
                double sigma = 0.05 + 0.5 * std::max(0.0, center);
                color sum;
                double weight_sum = 0;
                for (int ky = -2; ky <= 2; ++ky)
                    for (int kx = -2; kx <= 2; ++kx) {
                        int px = x + kx * step, py = y + ky * step;
                        if (px < 0 || py < 0 || px >= input.width || py >= input.height)
                            continue;
                        auto j = std::size_t(py) * input.width + px;
                        const auto &other = input.guides[j];
                        if (!other.hit || !other.filterable)
                            continue;
                        double normal = std::max(0.0, dot(guide.normal, other.normal));
                        double depth = std::fabs(guide.depth - other.depth) /
                                       (0.02 * std::max(0.1, guide.depth) * step);
                        double albedo = (guide.albedo - other.albedo).dot() / 0.02;
                        double light = std::fabs(center - luminance(output.linear[j])) / sigma;
                        double weight = kernel[kx + 2] * kernel[ky + 2] * std::pow(normal, 64) *
                                        std::exp(-depth - albedo - light);
                        sum += output.linear[j] * weight;
                        weight_sum += weight;
                    }
                next[i] = weight_sum > 0 ? sum / weight_sum : output.linear[i];
            }
        output.linear.swap(next);
    }
    return output;
}
