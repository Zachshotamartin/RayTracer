#include "feature_buffers.h"
#include "reconstruction.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#ifndef RT_HAS_ONNX
#define RT_HAS_ONNX 0
#endif
#if RT_HAS_ONNX
#include "../third_party/json.hpp"
#include <onnxruntime_cxx_api.h>
#endif

namespace {
[[maybe_unused]] bool same_view(const frame_snapshot &a, const frame_snapshot &b) {
    return (a.camera.lookfrom - b.camera.lookfrom).dot() == 0 &&
           (a.camera.lookat - b.camera.lookat).dot() == 0 &&
           (a.camera.vup - b.camera.vup).dot() == 0 && a.camera.vfov == b.camera.vfov;
}
vec3 feature_float(const vec3 &v) {
    return vec3(float(v.x()), float(v.y()), float(v.z()));
}
} // namespace
void append_reprojected_history(const frame_snapshot &current, const frame_snapshot &previous,
                                const std::vector<color> &rgb, std::vector<float> &input) {
    const auto size = current.linear.size();
    input.resize(size * 21, 0);
    if (current.history_key.empty() || current.history_key != previous.history_key ||
        current.width != previous.width || current.height != previous.height ||
        current.features.size() != size || previous.features.size() != size || rgb.size() != size ||
        (current.camera.lookfrom - previous.camera.lookfrom).length() > 3)
        return;
    auto w = to_unit_vector(previous.camera.lookfrom - previous.camera.lookat);
    auto u = to_unit_vector(cross(previous.camera.vup, w)), v = cross(w, u);
    double tangent = std::tan(degrees_to_radians(previous.camera.vfov) / 2);
    for (std::size_t i = 0; i < size; ++i) {
        const auto &g = current.features[i];
        if (float(g.support) < .999999f)
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
            input[(17 + c) * size + i] = float(rgb[index][c]);
        input[20 * size + i] = 1;
    }
}
struct neural_denoiser::impl {
    int output_scale = 1;
    bool temporal = false;
    std::optional<frame_snapshot> current, previous;
    std::vector<color> current_rgb, previous_rgb;
#if RT_HAS_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "RayTracer"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    impl(const std::filesystem::path &path, const std::string &provider) {
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (provider == "coreml") {
#ifdef __APPLE__
            options.AppendExecutionProvider(
                "CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}});
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
        if (!schema || std::string(schema.get()) != "1" || !scale ||
            (std::string(scale.get()) != "1" && std::string(scale.get()) != "2"))
            throw std::invalid_argument("Incompatible neural model metadata");
        output_scale = std::stoi(scale.get());
        auto history = metadata.LookupCustomMetadataMapAllocated("rt_temporal", allocator);
        temporal = history && std::string(history.get()) == "1";
        auto channel_text = metadata.LookupCustomMetadataMapAllocated("rt_channels", allocator);
        auto domain = metadata.LookupCustomMetadataMapAllocated("rt_domain", allocator);
        nlohmann::json expected = {
            "r",          "g",          "b",          "albedo_r", "albedo_g",      "albedo_b",
            "normal_x",   "normal_y",   "normal_z",   "depth",    "coverage",      "support",
            "variance_r", "variance_g", "variance_b", "samples",  "variance_valid"};
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
        if (std::string(input_name.get()) != "features" ||
            std::string(output_name.get()) != "radiance" ||
            input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 4 ||
            shape[1] != (temporal ? 21 : 17))
            throw std::invalid_argument("Incompatible neural tensor schema");
    }
#else
    impl(const std::filesystem::path &, const std::string &) {
        throw std::runtime_error("Neural inference unavailable: configure RAYTRACER_ENABLE_ONNX=ON "
                                 "with ONNXRUNTIME_ROOT");
    }
#endif
};
neural_denoiser::neural_denoiser(const std::filesystem::path &path, const std::string &provider)
    : impl_(std::make_unique<impl>(path, provider)) {}
neural_denoiser::~neural_denoiser() = default;
int neural_denoiser::scale() const {
    return impl_->output_scale;
}
frame_snapshot neural_denoiser::reconstruct(const frame_snapshot &frame) {
#if RT_HAS_ONNX
    if (std::uint64_t(frame.width) * frame.height * scale() * scale() > 16777216)
        throw std::invalid_argument("Neural output exceeds 16 megapixels");
    auto data = pack_reconstruction_input(frame);
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
            append_reprojected_history(frame, *impl_->previous, impl_->previous_rgb, data);
        else
            data.resize(frame.linear.size() * 21, 0);
    }
    std::array<int64_t, 4> dimensions{1, impl_->temporal ? 21 : 17, frame.height, frame.width};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input =
        Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), dimensions.data(), 4);
    const char *names[] = {"features"}, *outputs[] = {"radiance"};
    auto result = impl_->session->Run(Ort::RunOptions{nullptr}, names, &input, 1, outputs, 1);
    auto info = result[0].GetTensorTypeAndShapeInfo();
    if (info.GetShape() !=
            std::vector<int64_t>{1, 3, frame.height * scale(), frame.width * scale()} ||
        info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Unexpected neural output shape/type");
    frame_snapshot out{
        frame.width * scale(), frame.height * scale(), frame.samples,
        std::vector<color>(std::size_t(frame.width) * frame.height * scale() * scale())};
    const float *values = result[0].GetTensorData<float>();
    for (std::size_t i = 0; i < out.linear.size(); ++i)
        for (int c = 0; c < 3; ++c) {
            auto value = values[std::size_t(c) * out.linear.size() + i];
            if (!std::isfinite(value) || value < 0 || value > 1e30)
                throw std::runtime_error("Invalid neural radiance");
            out.linear[i][c] = value;
        }
    out.reconstructed = true;
    out.camera = frame.camera;
    out.history_key = frame.history_key;
    if (impl_->temporal) {
        impl_->current = frame;
        impl_->current_rgb = out.linear;
    }
    return out;
#else
    (void)frame;
    throw std::runtime_error("Neural inference was not compiled");
#endif
}
struct reconstruction_worker::impl {
    std::filesystem::path path;
    std::string provider;
    std::mutex mutex;
    std::condition_variable condition;
    bool stop = false;
    std::optional<std::pair<render_result, std::uint64_t>> pending;
    std::optional<reconstruction_result> completed;
    std::thread worker;
    impl(std::filesystem::path p, std::string device)
        : path(std::move(p)), provider(std::move(device)), worker([this] { run(); }) {}
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
            model = std::make_unique<neural_denoiser>(path, provider);
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
                                             const std::string &provider)
    : impl_(std::make_unique<impl>(p, provider)) {}
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
