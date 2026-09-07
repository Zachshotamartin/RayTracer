#include "feature_buffers.h"
#include "reconstruction.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#ifndef RT_HAS_ONNX
#define RT_HAS_ONNX 0
#endif
#if RT_HAS_ONNX
#include "../third_party/json.hpp"
#include <onnxruntime_cxx_api.h>
#endif

namespace {
[[maybe_unused]] bool same_view(const frame_snapshot &a, const frame_snapshot &b) {
    if (a.frame_index >= 0 || b.frame_index >= 0)
        return a.frame_index == b.frame_index;
    return (a.camera.lookfrom - b.camera.lookfrom).dot() == 0 &&
           (a.camera.lookat - b.camera.lookat).dot() == 0 &&
           (a.camera.vup - b.camera.vup).dot() == 0 && a.camera.vfov == b.camera.vfov;
}
vec3 feature_float(const vec3 &v) {
    return vec3(float(v.x()), float(v.y()), float(v.z()));
}
} // namespace
void append_reprojected_history(const frame_snapshot &current, const frame_snapshot &previous,
                                const std::vector<color> &rgb, std::vector<float> &input,
                                int schema) {
    const auto size = current.linear.size();
    const int channels = schema == 2 ? 27 : 17;
    input.resize(size * (channels + 4), 0);
    std::fill(input.begin() + channels * size, input.end(), 0);
    if (current.history_key.empty() || current.history_key != previous.history_key ||
        current.width != previous.width || current.height != previous.height ||
        current.features.size() != size || previous.features.size() != size || rgb.size() != size ||
        (schema == 1 && (current.camera.lookfrom - previous.camera.lookfrom).length() > 3))
        return;
    if (current.frame_index >= 0 && previous.frame_index >= 0 &&
        current.frame_index != previous.frame_index + 1)
        return;
    auto w = to_unit_vector(previous.camera.lookfrom - previous.camera.lookat);
    auto u = to_unit_vector(cross(previous.camera.vup, w)), v = cross(w, u);
    double tangent = std::tan(degrees_to_radians(previous.camera.vfov) / 2);
    if (schema == 2) {
        const double distance = (previous.camera.lookfrom - previous.camera.lookat).length();
        const auto forward = to_unit_vector(current.camera.lookfrom - current.camera.lookat);
        const auto current_u = to_unit_vector(cross(current.camera.vup, forward));
        if (distance < 1e-8 ||
            (current.camera.lookfrom - previous.camera.lookfrom).length() > .35 * distance ||
            dot(w, forward) < std::cos(degrees_to_radians(20)) ||
            dot(u, current_u) < std::cos(degrees_to_radians(20)) ||
            std::abs(std::log(current.camera.vfov / previous.camera.vfov)) > .18)
            return;
    }
    for (std::size_t i = 0; i < size; ++i) {
        const auto &g = current.features[i];
        if (schema == 1 && float(g.support) < .999999f)
            continue;
        if (schema == 2 && (float(g.center_support) <= .5f || float(g.center_depth) <= 0 ||
                            std::abs(float(g.coverage) - float(g.support)) >= 1e-6f))
            continue;
        auto position = feature_float(g.position), delta = position - previous.camera.lookfrom;
        double depth = -dot(delta, w);
        if (depth <= 0)
            continue;
        double px =
            (dot(delta, u) / (depth * tangent * (double(current.width) / current.height)) + 1) *
                current.width / 2 -
            .5;
        double py = (-dot(delta, v) / (depth * tangent) + 1) * current.height / 2 - .5;
        if (!std::isfinite(px) || !std::isfinite(py) || px < -.5 || py < -.5 ||
            px >= current.width - .5 || py >= current.height - .5)
            continue;
        if (schema == 2) {
            const int left = static_cast<int>(std::floor(px));
            const int top = static_cast<int>(std::floor(py));
            const double fx = px - left, fy = py - top;
            const double tolerance = std::max(1e-5, 3 * tangent * depth / current.height);
            const auto normal = feature_float(g.center_normal);
            const auto albedo = feature_float(g.center_albedo);
            color radiance(0, 0, 0);
            double total = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = left + dx, y = top + dy;
                    if (x < 0 || x >= current.width || y < 0 || y >= current.height)
                        continue;
                    const auto index = std::size_t(y) * current.width + x;
                    const auto &old = previous.features[index];
                    const auto old_normal = feature_float(old.center_normal);
                    const auto difference = feature_float(old.center_albedo) - albedo;
                    if (float(old.center_support) <= .5f ||
                        std::abs(float(old.coverage) - float(old.support)) >= 1e-6f ||
                        (feature_float(old.position) - position).length() >= tolerance ||
                        dot(normal, old_normal) /
                                std::max(1e-8, normal.length() * old_normal.length()) <=
                            .95 ||
                        std::max({std::abs(difference.x()), std::abs(difference.y()),
                                  std::abs(difference.z())}) >= .1)
                        continue;
                    const double weight = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
                    radiance += weight * rgb[index];
                    total += weight;
                }
            if (total >= .25) {
                for (int c = 0; c < 3; ++c)
                    input[(channels + c) * size + i] = float(radiance[c] / total);
                input[(channels + 3) * size + i] = float(total);
            }
            continue;
        }
        int x = static_cast<int>(std::floor(px + .5)), y = static_cast<int>(std::floor(py + .5));
        if (x < 0 || x >= current.width || y < 0 || y >= current.height)
            continue;
        auto index = std::size_t(y) * current.width + x;
        const auto &old = previous.features[index];
        if (float(old.support) < .999999f ||
            (feature_float(old.position) - position).length() >= .02 * std::max(1., delta.length()))
            continue;
        auto normal = feature_float(g.normal), old_normal = feature_float(old.normal);
        if (dot(normal, old_normal) / std::max(1e-8, normal.length() * old_normal.length()) <= .9)
            continue;
        for (int c = 0; c < 3; ++c)
            input[(channels + c) * size + i] = float(rgb[index][c]);
        input[(channels + 3) * size + i] = 1;
    }
}
struct neural_denoiser::impl {
    int output_scale = 1;
    int feature_schema = 1, base_channels = 17;
    std::vector<float> input_buffer;
    std::vector<float> tile_buffer;
    int tile_size = 0, tile_halo = 0, max_input_mib = 1024;
    neural_stats statistics;
    bool temporal = false;
    std::optional<frame_snapshot> current, previous;
    std::vector<color> current_rgb, previous_rgb;
#if RT_HAS_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "RayTracer"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    impl(const std::filesystem::path &path, const std::string &provider,
         const neural_settings &settings) {
        if (settings.threads < 1 || settings.threads > 64)
            throw std::invalid_argument("Neural threads must be 1..64");
        if ((settings.tile_size != 0 && (settings.tile_size < 16 || settings.tile_size > 2048)) ||
            settings.max_input_mib < 16 || settings.max_input_mib > 8192)
            throw std::invalid_argument("Invalid neural tile size or input memory limit");
        tile_size = settings.tile_size;
        max_input_mib = settings.max_input_mib;
        options.SetIntraOpNumThreads(settings.threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (!settings.profile_prefix.empty()) {
            if (!settings.profile_prefix.parent_path().empty())
                std::filesystem::create_directories(settings.profile_prefix.parent_path());
            options.EnableProfiling(settings.profile_prefix.c_str());
        }
        if (provider == "coreml") {
#ifdef __APPLE__
            std::unordered_map<std::string, std::string> coreml{{"ModelFormat", "MLProgram"},
                                                                {"MLComputeUnits", "ALL"}};
            if (!settings.cache_directory.empty()) {
                std::filesystem::create_directories(settings.cache_directory);
                coreml["ModelCacheDirectory"] = settings.cache_directory.string();
            }
            if (!settings.profile_prefix.empty())
                coreml["ProfileComputePlan"] = "1";
            options.AppendExecutionProvider("CoreML", coreml);
#else
            throw std::invalid_argument("Core ML requires macOS");
#endif
        } else if (provider != "cpu")
            throw std::invalid_argument("Provider must be cpu or coreml");
        session = std::make_unique<Ort::Session>(env, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        auto metadata = session->GetModelMetadata();
        auto schema = metadata.LookupCustomMetadataMapAllocated("rt_schema", allocator);
        auto scale = metadata.LookupCustomMetadataMapAllocated("rt_scale", allocator);
        if (!schema || (std::string(schema.get()) != "1" && std::string(schema.get()) != "2") ||
            !scale || (std::string(scale.get()) != "1" && std::string(scale.get()) != "2"))
            throw std::invalid_argument("Incompatible neural model metadata");
        output_scale = std::stoi(scale.get());
        feature_schema = std::stoi(schema.get());
        base_channels = feature_schema == 2 ? 27 : 17;
        auto history = metadata.LookupCustomMetadataMapAllocated("rt_temporal", allocator);
        temporal = history && std::string(history.get()) == "1";
        auto channel_text = metadata.LookupCustomMetadataMapAllocated("rt_channels", allocator);
        auto domain = metadata.LookupCustomMetadataMapAllocated("rt_domain", allocator);
        nlohmann::json expected = {
            "r",          "g",          "b",          "albedo_r", "albedo_g",      "albedo_b",
            "normal_x",   "normal_y",   "normal_z",   "depth",    "coverage",      "support",
            "variance_r", "variance_g", "variance_b", "samples",  "variance_valid"};
        if (feature_schema == 2)
            for (const auto *name :
                 {"center_albedo_r", "center_albedo_g", "center_albedo_b", "center_normal_x",
                  "center_normal_y", "center_normal_z", "center_depth", "depth_variance",
                  "normal_spread", "center_support"})
                expected.push_back(name);
        if (temporal)
            for (const auto *name : {"history_r", "history_g", "history_b", "history_valid"})
                expected.push_back(name);
        if (!channel_text || nlohmann::json::parse(channel_text.get()) != expected || !domain ||
            std::string(domain.get()) != "diffuse-pinhole" ||
            (history && std::string(history.get()) != "0" && std::string(history.get()) != "1"))
            throw std::invalid_argument("Incompatible neural channel/domain metadata");
        if (temporal && output_scale != 1)
            throw std::invalid_argument("Temporal model requires scale 1");
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
            throw std::invalid_argument(
                "Model must have one feature input and one radiance output");
        auto input_name = session->GetInputNameAllocated(0, allocator);
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        auto input_type = session->GetInputTypeInfo(0);
        auto input = input_type.GetTensorTypeAndShapeInfo();
        auto shape = input.GetShape();
        auto halo = metadata.LookupCustomMetadataMapAllocated("rt_tile_halo", allocator);
        if (halo) {
            tile_halo = std::stoi(halo.get());
            if (tile_halo < 0 || tile_halo > 64)
                throw std::invalid_argument("Invalid model tile halo");
        }
        if (std::string(input_name.get()) != "features" ||
            std::string(output_name.get()) != "radiance" ||
            input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 4 ||
            shape[1] != base_channels + (temporal ? 4 : 0))
            throw std::invalid_argument("Incompatible neural tensor schema");
        if (shape[2] > 0 || shape[3] > 0)
            tile_halo = 0;
        if (tile_size && !tile_halo)
            throw std::invalid_argument(
                "Tiling requires a dynamic model with a declared local receptive field");
    }
#else
    impl(const std::filesystem::path &, const std::string &, const neural_settings &) {
        throw std::runtime_error("Neural inference unavailable: configure RAYTRACER_ENABLE_ONNX=ON "
                                 "with ONNXRUNTIME_ROOT");
    }
#endif
};
neural_denoiser::neural_denoiser(const std::filesystem::path &path, const std::string &provider,
                                 neural_settings settings)
    : impl_(std::make_unique<impl>(path, provider, settings)) {}
neural_denoiser::~neural_denoiser() = default;
int neural_denoiser::scale() const {
    return impl_->output_scale;
}
neural_stats neural_denoiser::stats() const {
    return impl_->statistics;
}
frame_snapshot neural_denoiser::reconstruct(const frame_snapshot &frame) {
#if RT_HAS_ONNX
    if (std::uint64_t(frame.width) * frame.height * scale() * scale() > 16777216)
        throw std::invalid_argument("Neural output exceeds 16 megapixels");
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    impl_->statistics = {};
    const int channels = impl_->base_channels + (impl_->temporal ? 4 : 0);
    if (std::uint64_t(frame.width) * frame.height * channels * sizeof(float) >
        std::uint64_t(impl_->max_input_mib) * 1024 * 1024)
        throw std::invalid_argument("Neural input exceeds its configured memory limit");
    auto &data = impl_->input_buffer;
    pack_reconstruction_input(frame, data, impl_->feature_schema);
    if (impl_->temporal) {
        if (impl_->current &&
            (impl_->current->history_key != frame.history_key ||
             impl_->current->width != frame.width || impl_->current->height != frame.height)) {
            impl_->current.reset();
            impl_->previous.reset();
        }
        if (impl_->current && !same_view(frame, *impl_->current)) {
            impl_->previous = std::move(impl_->current);
            impl_->current.reset();
            impl_->previous_rgb = std::move(impl_->current_rgb);
        }
        if (impl_->previous)
            append_reprojected_history(frame, *impl_->previous, impl_->previous_rgb, data,
                                       impl_->feature_schema);
        else
            data.resize(frame.linear.size() * (impl_->base_channels + 4), 0);
    }
    impl_->statistics.input_bytes = data.size() * sizeof(float);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char *names[] = {"features"}, *outputs[] = {"radiance"};
    const auto packed = clock::now();
    impl_->statistics.packing_seconds = std::chrono::duration<double>(packed - start).count();
    frame_snapshot out{
        frame.width * scale(), frame.height * scale(), frame.samples,
        std::vector<color>(std::size_t(frame.width) * frame.height * scale() * scale())};
    impl_->statistics.output_seconds +=
        std::chrono::duration<double>(clock::now() - packed).count();
    int tile = impl_->tile_size;
    if (!tile && impl_->tile_halo && frame.linear.size() > 512 * 512)
        tile = 256;
    if (!tile)
        tile = std::max(frame.width, frame.height);
    for (int top = 0; top < frame.height; top += tile)
        for (int left = 0; left < frame.width; left += tile) {
            const auto copy_start = clock::now();
            const int right = std::min(frame.width, left + tile);
            const int bottom = std::min(frame.height, top + tile);
            const int x0 = std::max(0, left - impl_->tile_halo);
            const int y0 = std::max(0, top - impl_->tile_halo);
            const int x1 = std::min(frame.width, right + impl_->tile_halo);
            const int y1 = std::min(frame.height, bottom + impl_->tile_halo);
            const int width = x1 - x0, height = y1 - y0;
            const auto pixels = std::size_t(width) * height;
            const bool whole = width == frame.width && height == frame.height;
            float *input_data = data.data();
            if (!whole) {
                auto &buffer = impl_->tile_buffer;
                buffer.resize(pixels * channels);
                for (int c = 0; c < channels; ++c)
                    for (int y = 0; y < height; ++y)
                        std::copy_n(data.data() + std::size_t(c) * frame.linear.size() +
                                        std::size_t(y0 + y) * frame.width + x0,
                                    width,
                                    buffer.data() + std::size_t(c) * pixels +
                                        std::size_t(y) * width);
                input_data = buffer.data();
            }
            std::array<int64_t, 4> dimensions{1, channels, height, width};
            auto input = Ort::Value::CreateTensor<float>(memory, input_data, pixels * channels,
                                                         dimensions.data(), 4);
            const auto infer_start = clock::now();
            impl_->statistics.packing_seconds +=
                std::chrono::duration<double>(infer_start - copy_start).count();
            auto result =
                impl_->session->Run(Ort::RunOptions{nullptr}, names, &input, 1, outputs, 1);
            const auto inferred = clock::now();
            impl_->statistics.inference_seconds +=
                std::chrono::duration<double>(inferred - infer_start).count();
            auto info = result[0].GetTensorTypeAndShapeInfo();
            if (info.GetShape() != std::vector<int64_t>{1, 3, height * scale(), width * scale()} ||
                info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Unexpected neural output shape/type");
            const float *values = result[0].GetTensorData<float>();
            for (int y = top * scale(); y < bottom * scale(); ++y)
                for (int x = left * scale(); x < right * scale(); ++x) {
                    const auto source =
                        std::size_t(y - y0 * scale()) * width * scale() + x - x0 * scale();
                    const auto destination = std::size_t(y) * out.width + x;
                    for (int c = 0; c < 3; ++c) {
                        const auto value =
                            values[std::size_t(c) * pixels * scale() * scale() + source];
                        if (!std::isfinite(value) || value < 0 || value > 1e30)
                            throw std::runtime_error("Invalid neural radiance");
                        out.linear[destination][c] = value;
                    }
                }
            impl_->statistics.output_seconds +=
                std::chrono::duration<double>(clock::now() - inferred).count();
            ++impl_->statistics.tiles;
        }
    impl_->statistics.input_bytes =
        (impl_->input_buffer.capacity() + impl_->tile_buffer.capacity()) * sizeof(float);
    const auto history_start = clock::now();
    out.reconstructed = true;
    out.camera = frame.camera;
    out.frame_index = frame.frame_index;
    out.history_key = frame.history_key;
    if (impl_->temporal) {
        impl_->current = frame;
        impl_->current_rgb = out.linear;
    }
    impl_->statistics.output_seconds +=
        std::chrono::duration<double>(clock::now() - history_start).count();
    return out;
#else
    (void)frame;
    throw std::runtime_error("Neural inference was not compiled");
#endif
}
struct reconstruction_worker::impl {
    std::filesystem::path path;
    std::string provider;
    neural_settings settings;
    std::mutex mutex;
    std::condition_variable condition;
    bool stop = false;
    std::optional<std::pair<render_result, std::uint64_t>> pending;
    std::optional<reconstruction_result> completed;
    std::thread worker;
    impl(std::filesystem::path p, std::string device, neural_settings config)
        : path(std::move(p)), provider(std::move(device)), settings(std::move(config)),
          worker([this] { run(); }) {}
    ~impl() {
        {
            std::lock_guard lock(mutex);
            stop = true;
            pending.reset();
        }
        condition.notify_all();
        worker.join();
    }
    void run() {
        std::unique_ptr<neural_denoiser> model;
        std::string load_error;
        try {
            model = std::make_unique<neural_denoiser>(path, provider, settings);
        } catch (const std::exception &e) {
            load_error = e.what();
        }
        for (;;) {
            std::pair<render_result, std::uint64_t> request;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return stop || pending.has_value(); });
                if (stop)
                    return;
                request = std::move(*pending);
                pending.reset();
            }
            auto start = std::chrono::steady_clock::now();
            reconstruction_result result;
            result.generation = request.second;
            try {
                if (!load_error.empty())
                    throw std::runtime_error(load_error);
                result.frame = model->reconstruct(request.first.frame);
                result.details = model->stats();
            } catch (const std::exception &e) {
                result.error = e.what();
                result.frame = request.first.frame;
            }
            result.raw = std::move(request.first.frame);
            result.stats = request.first.stats;
            result.seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            {
                std::lock_guard lock(mutex);
                completed = std::move(result);
            }
        }
    }
};
reconstruction_worker::reconstruction_worker(const std::filesystem::path &p,
                                             const std::string &provider, neural_settings settings)
    : impl_(std::make_unique<impl>(p, provider, settings)) {}
reconstruction_worker::~reconstruction_worker() = default;
void reconstruction_worker::submit(const frame_snapshot &f, const render_stats &stats,
                                   std::uint64_t generation) {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->pending = std::make_pair(render_result{f, stats}, generation);
    }
    impl_->condition.notify_one();
}
std::optional<reconstruction_result> reconstruction_worker::poll() {
    std::lock_guard lock(impl_->mutex);
    auto result = std::move(impl_->completed);
    impl_->completed.reset();
    return result;
}
