#include "../third_party/json.hpp"
#include "denoiser.h"
#include "feature_buffers.h"
#include "reconstruction.h"
#include "renderer.h"
#include "scene_file.h"
#ifndef RT_HAS_SDL
#define RT_HAS_SDL 1
#endif
#if RT_HAS_SDL
#include "sdl_window.h"
#endif
#include <charconv>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt_handler(int) {
    interrupted = 1;
}
struct options {
    render_settings render;
    std::string scene = "studio";
    std::filesystem::path output;
    std::filesystem::path mesh;
    std::filesystem::path scene_file, write_scene, feature_directory;
    std::optional<std::uint64_t> scene_seed;
    std::filesystem::path model, raw_output, reference_file;
    std::string scene_description;
    std::filesystem::path sequence_directory, sequence_output, sequence_raw_output;
    std::vector<std::filesystem::path> sequence_files;
    std::size_t sequence_index = 0;
    std::uint64_t sequence_seed = 42;
    std::string neural_provider = "cpu", neural_error;
    neural_settings neural_config;
    neural_stats neural_details;
    int neural_interval_ms = 150;
    bool neural_active = false, reconstructed = false, neural_supported = true;
    double neural_seconds = 0, model_load_seconds = 0, pipeline_seconds = 0;
    int benchmark_repeats = 1, benchmark_iteration = 0;
    std::chrono::steady_clock::time_point pipeline_start = std::chrono::steady_clock::now();
    std::filesystem::path default_output_directory = "renders";
    double exposure = 0;
    bool headless = !RT_HAS_SDL, quiet = false, quit_after_render = false;
    bool denoise = false;
    double denoise_seconds = 0, image_write_seconds = 0;
};
double real(const std::string &text) {
    std::size_t used = 0;
    double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected a finite number, got '" + text + "'");
    return value;
}
scene build_scene(const options &opts) {
    if (!opts.scene_file.empty())
        return read_scene_file(opts.scene_file);
    if (opts.mesh.empty())
        return make_scene(opts.scene, opts.scene_seed.value_or(opts.render.seed));
    std::string warnings;
    auto result = make_mesh_scene(opts.mesh, warnings);
    if (!warnings.empty())
        std::cerr << "OBJ: " << warnings;
    return result;
}
std::string scene_description(const options &opts) {
    if (!opts.scene_file.empty()) {
        std::ifstream in(opts.scene_file);
        return nlohmann::json::parse(in).dump();
    }
    if (!opts.mesh.empty())
        return "mesh-reference-unsupported";
    return opts.scene + ":" +
           std::to_string(opts.scene == "field" ? opts.scene_seed.value_or(opts.render.seed) : 0);
}
std::string history_description(const options &opts) {
    if (!opts.scene_file.empty()) {
        auto j = nlohmann::json::parse(opts.scene_description);
        j.erase("camera");
        j.erase("name");
        return j.dump() + ":" + std::to_string(opts.render.max_depth);
    }
    return opts.scene_description + ":" + std::to_string(opts.render.max_depth);
}
void select_sequence_frame(options &opts, std::size_t index) {
    opts.sequence_index = index;
    opts.scene_file = opts.sequence_files.at(index);
    opts.render.seed = opts.sequence_seed + index;
    auto numbered = [&](const std::filesystem::path &base) {
        std::ostringstream number;
        number << base.stem().string() << '-' << std::setfill('0') << std::setw(6) << index
               << base.extension().string();
        return base.parent_path() / number.str();
    };
    opts.output = numbered(opts.sequence_output);
    if (!opts.sequence_raw_output.empty())
        opts.raw_output = numbered(opts.sequence_raw_output);
}
bool last_sequence_frame(const options &opts) {
    return opts.sequence_files.empty() || opts.sequence_index + 1 == opts.sequence_files.size();
}
std::string fingerprint(const options &opts, int width, int height) {
    // Stable non-security identity for matching reference content, independent of noise seed.
    nlohmann::json j = {opts.scene_description,
                        width,
                        height,
                        opts.render.max_depth,
                        opts.render.caustic_photons,
                        opts.render.caustic_radius,
                        opts.render.transparent_shadows,
                        "path-mis-ggx-v2"};
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : j.dump()) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}
std::string quote(const std::string &value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') {
            out += '\\';
            out += c;
        } else if (c < 32) {
            const char *hex = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else
            out += c;
    }
    return out + '"';
}
template <class T> T integer(const std::string &value) {
    T number{};
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument("Expected an integer, got '" + value + "'");
    return number;
}
void help() {
    std::cout
        << "RayTracer - progressive CPU path tracer\n"
           "Usage: raytracer [--headless] [options]\n"
           "  --scene-file PATH           Versioned JSON scene/camera configuration\n"
           "  --sequence DIRECTORY        Play sorted JSON scene frames; headless exports each\n"
           "  --scene-seed N              Scene distribution seed independent of noise\n"
           "  --write-scene PATH          Export reproducible scene configuration\n"
           "  --features DIRECTORY        Export aligned linear training buffers (headless)\n"
           "  --model PATH.onnx           Reconstruct with trained weights; N toggles in viewer\n"
           "  --neural-provider NAME      cpu (default) or coreml\n"
           "  --neural-threads N          Inference CPU threads (default 2)\n"
           "  --neural-cache DIR          Cache compiled Core ML models\n"
           "  --neural-profile PATH       Write ORT profile with this prefix\n"
           "  --neural-interval-ms N      Minimum time between preview requests (default 150)\n"
           "  --raw-output PATH.pfm       Also export unchanged raw accumulation\n"
           "  --reference PATH.pfm        Raw reference + JSON metadata; V/E compare in viewer\n"
           "  --benchmark-repeats N       Repeat headless frames with one loaded model\n"
           "  --scene NAME                demo, field, studio, or caustics\n"
           "  --mesh PATH.obj             Import OBJ/MTL into a lit studio\n"
           "  --width N                   Image width (default 640)\n"
           "  --samples N                 Samples per pixel (default 64)\n"
           "  --depth N                   Maximum path vertices (default 16)\n"
           "  --seed N                    Reproducible scene/path seed (default 42)\n"
           "  --threads N                 CPU workers; 0 selects hardware count\n"
           "  --no-bvh                    Use linear object traversal\n"
           "  --output PATH               .png, .hdr, or .pfm plus JSON metadata\n"
           "  --denoise                   Edge-aware diffuse surface filtering\n"
           "  --caustics N                Trace N caustic photons (default off)\n"
           "  --caustic-radius R          Gather radius in world units (default 0.12)\n"
           "  --glass-shadows MODE        physical (default) or transparent approximation\n"
           "  --exposure EV               Display exposure, -8 through 8 stops\n"
           "  --quiet                     Suppress headless progress\n"
           "  --quit-after-render         Close viewer after export\n"
           "  --list-scenes               Print preset names\n"
           "Headless mode writes renders/SCENE.png by default and reports JSON on stdout.\n"
           "Viewer: 1/2/3/4 scenes, D denoise, N AI, V reference, E error, C caustics, G glass "
           "shadows, Space pause, R "
           "restart, S save, Esc cancel, Q quit, +/- "
           "exposure.\n";
}
std::string report(const options &opts, const frame_snapshot &frame, const render_stats &stats) {
    std::ostringstream out;
    out << std::setprecision(10) << "{\n  \"scene\": " << quote(opts.scene)
        << ",\n  \"seed\": " << opts.render.seed << ",\n  \"width\": " << frame.width
        << ",\n  \"height\": " << frame.height << ",\n  \"samples\": " << frame.samples
        << ",\n  \"requested_samples\": " << opts.render.samples
        << ",\n  \"max_depth\": " << opts.render.max_depth << ",\n  \"workers\": " << stats.workers
        << ",\n  \"bvh\": " << (opts.render.use_bvh ? "true" : "false")
        << ",\n  \"objects\": " << stats.objects << ",\n  \"render_seconds\": " << stats.seconds
        << ",\n  \"feature_rays\": " << stats.feature_rays
        << ",\n  \"scene_file\": " << quote(opts.scene_file.generic_string())
        << ",\n  \"scene_fingerprint\": " << quote(fingerprint(opts, frame.width, frame.height))
        << ",\n  \"scene_seed\": "
        << (opts.scene_file.empty() ? std::to_string(opts.scene_seed.value_or(opts.render.seed))
                                    : "null")
        << ",\n  \"bvh_build_seconds\": " << stats.build_seconds
        << ",\n  \"path_rays\": " << stats.path_rays
        << ",\n  \"shadow_rays\": " << stats.shadow_rays << ",\n  \"exposure\": " << opts.exposure
        << ",\n  \"mesh_path\": " << quote(opts.mesh.generic_string())
        << ",\n  \"denoised\": " << (opts.denoise && !opts.reconstructed ? "true" : "false")
        << ",\n  \"reconstructed\": " << (opts.reconstructed ? "true" : "false")
        << ",\n  \"model\": " << quote(opts.model.generic_string())
        << ",\n  \"neural_provider_requested\": " << quote(opts.neural_provider)
        << ",\n  \"neural_error\": " << quote(opts.neural_error)
        << ",\n  \"neural_seconds\": " << opts.neural_seconds
        << ",\n  \"neural_packing_seconds\": " << opts.neural_details.packing_seconds
        << ",\n  \"neural_inference_seconds\": " << opts.neural_details.inference_seconds
        << ",\n  \"neural_output_seconds\": " << opts.neural_details.output_seconds
        << ",\n  \"neural_input_bytes\": " << opts.neural_details.input_bytes
        << ",\n  \"neural_threads\": " << opts.neural_config.threads
        << ",\n  \"model_load_seconds\": " << opts.model_load_seconds
        << ",\n  \"pipeline_seconds\": " << opts.pipeline_seconds
        << ",\n  \"benchmark_iteration\": " << opts.benchmark_iteration
        << ",\n  \"denoise_seconds\": " << opts.denoise_seconds
        << ",\n  \"image_write_seconds\": " << opts.image_write_seconds
        << ",\n  \"glass_shadows\": "
        << quote(opts.render.transparent_shadows ? "transparent-approximation" : "physical")
        << ",\n  \"caustic_photons\": " << opts.render.caustic_photons
        << ",\n  \"caustic_radius\": " << opts.render.caustic_radius
        << ",\n  \"emitted_photons\": " << stats.emitted_photons
        << ",\n  \"stored_photons\": " << stats.stored_photons
        << ",\n  \"photon_rays\": " << stats.photon_rays
        << ",\n  \"photon_seconds\": " << stats.photon_seconds << ",\n  \"output_format\": "
        << quote(opts.output.empty() ? ".png" : opts.output.extension().string())
        << ",\n  \"sampler\": \"splitmix64-per-pixel-v1\",\n  \"integrator\": \"path-mis-ggx-v2\","
           "\n  \"display\": \"aces-fit-srgb\",\n  \"hdr_encoding\": \"linear-no-exposure\"\n}\n";
    return out.str();
}
std::filesystem::path save(options &opts, const frame_snapshot &frame, const render_stats &stats,
                           const frame_snapshot *prediction = nullptr) {
    auto output =
        opts.output.empty() ? opts.default_output_directory / (opts.scene + ".png") : opts.output;
    if (frame.samples == 0)
        throw std::runtime_error("No complete sample pass is available to save");
    auto start = std::chrono::steady_clock::now();
    frame_snapshot filtered;
    if (!opts.raw_output.empty()) {
        if (std::filesystem::absolute(opts.raw_output).lexically_normal() ==
            std::filesystem::absolute(output).lexically_normal())
            throw std::invalid_argument("Raw and processed outputs must differ");
        write_image(opts.raw_output, frame, opts.exposure);
    }
    if (opts.denoise && !prediction)
        filtered = denoise(frame);
    auto processed = std::chrono::steady_clock::now();
    opts.denoise_seconds =
        opts.denoise && !prediction ? std::chrono::duration<double>(processed - start).count() : 0;
    opts.reconstructed = prediction && prediction->reconstructed;
    const auto &image = prediction ? *prediction : opts.denoise ? filtered : frame;
    write_image(output, image, opts.exposure);
    opts.image_write_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - processed).count();
    opts.pipeline_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - opts.pipeline_start)
            .count();
    auto metadata = output;
    metadata.replace_extension(".json");
    std::ofstream json(metadata);
    json << report(opts, image, stats);
    if (!json)
        throw std::runtime_error("Could not write metadata: " + metadata.string());
    return output;
}
} // namespace
int main(int argc, char **argv) {
    try {
        options opts;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc)
                    throw std::invalid_argument("Missing value after " + arg);
                return std::string(argv[++i]);
            };
            if (arg == "--help" || arg == "-h") {
                help();
                return 0;
            } else if (arg == "--list-scenes") {
                for (const auto &name : scene_names())
                    std::cout << name << '\n';
                return 0;
            } else if (arg == "--scene")
                opts.scene = value();
            else if (arg == "--scene-file")
                opts.scene_file = value();
            else if (arg == "--sequence")
                opts.sequence_directory = value();
            else if (arg == "--scene-seed")
                opts.scene_seed = integer<std::uint64_t>(value());
            else if (arg == "--write-scene")
                opts.write_scene = value();
            else if (arg == "--features")
                opts.feature_directory = value();
            else if (arg == "--model") {
                opts.model = value();
                opts.neural_active = true;
            } else if (arg == "--neural-provider")
                opts.neural_provider = value();
            else if (arg == "--neural-threads")
                opts.neural_config.threads = integer<int>(value());
            else if (arg == "--neural-cache")
                opts.neural_config.cache_directory = value();
            else if (arg == "--neural-profile")
                opts.neural_config.profile_prefix = value();
            else if (arg == "--neural-interval-ms") {
                opts.neural_interval_ms = integer<int>(value());
                if (opts.neural_interval_ms < 0 || opts.neural_interval_ms > 10000)
                    throw std::invalid_argument("Neural preview interval must be 0..10000 ms");
            } else if (arg == "--raw-output")
                opts.raw_output = value();
            else if (arg == "--reference")
                opts.reference_file = value();
            else if (arg == "--benchmark-repeats")
                opts.benchmark_repeats = integer<int>(value());
            else if (arg == "--width")
                opts.render.width = integer<int>(value());
            else if (arg == "--samples")
                opts.render.samples = integer<int>(value());
            else if (arg == "--depth")
                opts.render.max_depth = integer<int>(value());
            else if (arg == "--seed")
                opts.render.seed = integer<std::uint64_t>(value());
            else if (arg == "--threads")
                opts.render.threads = integer<int>(value());
            else if (arg == "--output")
                opts.output = value();
            else if (arg == "--mesh")
                opts.mesh = value();
            else if (arg == "--denoise")
                opts.denoise = true;
            else if (arg == "--caustics")
                opts.render.caustic_photons = integer<int>(value());
            else if (arg == "--caustic-radius")
                opts.render.caustic_radius = real(value());
            else if (arg == "--glass-shadows") {
                auto mode = value();
                if (mode != "physical" && mode != "transparent")
                    throw std::invalid_argument("Glass shadows must be physical or transparent");
                opts.render.transparent_shadows = mode == "transparent";
            } else if (arg == "--exposure") {
                auto input = value();
                std::size_t consumed = 0;
                opts.exposure = std::stod(input, &consumed);
                if (consumed != input.size() || !std::isfinite(opts.exposure) ||
                    std::fabs(opts.exposure) > 8)
                    throw std::invalid_argument("Exposure must be finite and between -8 and 8");
            } else if (arg == "--headless")
                opts.headless = true;
#ifdef __APPLE__
            // LaunchServices and XCTest append Cocoa defaults to GUI launches.
            else if (arg == "-NSTreatUnknownArgumentsAsOpen" ||
                     arg == "-ApplePersistenceIgnoreState")
                value();
            else if (arg.rfind("-psn_", 0) == 0) {
            }
#endif
            else if (arg == "--no-bvh")
                opts.render.use_bvh = false;
            else if (arg == "--quiet")
                opts.quiet = true;
            else if (arg == "--quit-after-render")
                opts.quit_after_render = true;
            else
                throw std::invalid_argument("Unknown option: " + arg);
        }
        if (!opts.sequence_directory.empty()) {
            if (!opts.scene_file.empty() || !opts.mesh.empty() || opts.benchmark_repeats != 1 ||
                !opts.feature_directory.empty())
                throw std::invalid_argument("Sequence mode is separate from single-scene, feature "
                                            "export, and repeated-frame benchmarks");
            for (const auto &entry : std::filesystem::directory_iterator(opts.sequence_directory))
                if (entry.is_regular_file() && entry.path().extension() == ".json")
                    opts.sequence_files.push_back(entry.path());
            std::sort(opts.sequence_files.begin(), opts.sequence_files.end());
            if (opts.sequence_files.empty() || opts.sequence_files.size() > 10000)
                throw std::invalid_argument("Sequence needs 1..10000 JSON frames");
            opts.sequence_output =
                opts.output.empty() ? opts.default_output_directory / "sequence.png" : opts.output;
            opts.sequence_raw_output = opts.raw_output;
            opts.sequence_seed = opts.render.seed;
            select_sequence_frame(opts, 0);
        }
        if (!opts.output.empty() && opts.output.extension() != ".png" &&
            opts.output.extension() != ".hdr" && opts.output.extension() != ".pfm")
            throw std::invalid_argument("Output must end in .png, .hdr, or .pfm");
        if (!opts.mesh.empty())
            opts.scene = "mesh";
        if (opts.benchmark_repeats < 1 || opts.benchmark_repeats > 100 ||
            (opts.benchmark_repeats > 1 && !opts.headless))
            throw std::invalid_argument("Benchmark repeats require 1..100 headless frames");
        if (!opts.feature_directory.empty() && !opts.headless)
            throw std::invalid_argument("Feature export requires --headless");
        if (!opts.scene_file.empty() && !opts.mesh.empty())
            throw std::invalid_argument("Choose --scene-file or --mesh");
        if (!opts.raw_output.empty() && opts.raw_output == opts.output)
            throw std::invalid_argument("Raw and reconstructed output paths must differ");
        opts.render.collect_features = !opts.feature_directory.empty() || !opts.model.empty();
        opts.render.collect_guides = opts.denoise || !opts.headless || opts.render.collect_features;
        auto content = build_scene(opts);
        opts.scene = content.name;
        opts.scene_description = scene_description(opts);
        opts.render.history_key = history_description(opts);
        opts.neural_supported = content.view.defocus_angle == 0 && !opts.render.caustic_photons &&
                                !opts.render.transparent_shadows;
        if (!opts.write_scene.empty()) {
            if (!opts.mesh.empty())
                throw std::invalid_argument(
                    "Mesh configuration export requires an explicit scene file");
            if (!opts.scene_file.empty()) {
                if (!opts.write_scene.parent_path().empty())
                    std::filesystem::create_directories(opts.write_scene.parent_path());
                if (!opts.write_scene.parent_path().empty())
                    std::filesystem::create_directories(opts.write_scene.parent_path());
                std::filesystem::copy_file(opts.scene_file, opts.write_scene,
                                           std::filesystem::copy_options::overwrite_existing);
            } else
                write_scene_file(opts.write_scene, opts.scene,
                                 opts.scene_seed.value_or(opts.render.seed), content.view);
        }
        opts.render.validate(content.view.aspect_ratio);
        std::signal(SIGINT, interrupt_handler);
        std::signal(SIGTERM, interrupt_handler);
        if (opts.headless) {
            std::unique_ptr<neural_denoiser> model;
            if (opts.neural_active) {
                auto start = std::chrono::steady_clock::now();
                try {
                    model = std::make_unique<neural_denoiser>(opts.model, opts.neural_provider,
                                                              opts.neural_config);
                } catch (const std::exception &error) {
                    opts.neural_error = error.what();
                    std::cerr << "Neural fallback: " << error.what() << '\n';
                }
                opts.model_load_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            }
            for (int iteration = 0;
                 iteration < (opts.sequence_files.empty() ? opts.benchmark_repeats
                                                          : int(opts.sequence_files.size()));
                 ++iteration) {
                if (!opts.sequence_files.empty()) {
                    select_sequence_frame(opts, iteration);
                    content = build_scene(opts);
                    opts.scene = content.name;
                    opts.scene = content.name;
                    opts.scene_description = scene_description(opts);
                    opts.render.history_key = history_description(opts);
                    opts.neural_supported = content.view.defocus_angle == 0 &&
                                            !opts.render.caustic_photons &&
                                            !opts.render.transparent_shadows;
                }
                opts.benchmark_iteration = iteration;
                opts.pipeline_start = std::chrono::steady_clock::now();
                render_session session(content, opts.render);
                auto last = std::chrono::steady_clock::now();
                while (!session.done()) {
                    if (interrupted)
                        session.cancel();
                    auto now = std::chrono::steady_clock::now();
                    if (!opts.quiet && now - last > std::chrono::milliseconds(500)) {
                        auto stats = session.stats();
                        std::cerr << "\r" << opts.scene << ": " << stats.samples << "/"
                                  << opts.render.samples << " spp" << std::flush;
                        last = now;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                session.wait();
                auto frame = session.snapshot();
                auto stats = session.stats();
                if (frame.samples > 0 && !opts.feature_directory.empty())
                    write_feature_buffers(opts.feature_directory, frame);
                frame_snapshot prediction;
                if (model && frame.samples > 0) {
                    auto start = std::chrono::steady_clock::now();
                    try {
                        if (!opts.neural_supported)
                            throw std::runtime_error("Unsupported model domain for this frame");
                        prediction = model->reconstruct(frame);
                        opts.neural_details = model->stats();
                        opts.neural_error.clear();
                    } catch (const std::exception &error) {
                        opts.neural_error = error.what();
                        std::cerr << "Neural fallback: " << error.what() << '\n';
                    }
                    opts.neural_seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                            .count();
                }
                if (frame.samples > 0)
                    save(opts, frame, stats, prediction.reconstructed ? &prediction : nullptr);
                if (!opts.quiet)
                    std::cerr << "\r" << opts.scene << ": " << frame.samples << " spp in "
                              << stats.seconds << " s\n";
                std::cout << report(opts, prediction.reconstructed ? prediction : frame, stats);
                if (interrupted)
                    break;
            }
            return interrupted ? 130 : 0;
        }
#if RT_HAS_SDL
        sdl_window window(opts.render.width, opts.render.height(content.view.aspect_ratio));
        auto working_directory = std::filesystem::current_path();
        if (working_directory == working_directory.root_path())
            opts.default_output_directory = window.preferred_output_directory();
        opts.pipeline_start = std::chrono::steady_clock::now();
        auto session = std::make_unique<render_session>(std::move(content), opts.render);
        frame_snapshot frame = session->snapshot();
        frame_snapshot preview = frame;
        std::unique_ptr<reconstruction_worker> neural_worker;
        if (!opts.model.empty())
            neural_worker = std::make_unique<reconstruction_worker>(
                opts.model, opts.neural_provider, opts.neural_config);
        std::optional<reconstruction_result> neural_preview;
        std::uint64_t generation = 0;
        int requested_samples = 0, save_target = 0;
        auto neural_request_time = std::chrono::steady_clock::time_point{};
        bool save_pending = false;
        int comparison = 0;
        bool playback_paused = false;
        frame_snapshot reference;
        std::string reference_identity, reference_error;
        if (!opts.reference_file.empty()) {
            try {
                reference = read_pfm(opts.reference_file);
                auto meta = opts.reference_file;
                meta.replace_extension(".json");
                std::ifstream in(meta);
                auto j = nlohmann::json::parse(in);
                if (j.value("denoised", false) || j.value("reconstructed", false) ||
                    !opts.mesh.empty())
                    throw std::invalid_argument("Reference must be a raw scene-file/preset render");
                reference_identity = j.at("scene_fingerprint").get<std::string>();
            } catch (const std::exception &e) {
                reference_error = e.what();
                std::cerr << "Reference: " << e.what() << '\n';
            }
        }
        auto preview_time = std::chrono::steady_clock::now();
        std::string notice;
        bool quit = false, exported = false, cancelled = false, completion_handled = false;
        while (!quit && !interrupted) {
            auto actions = window.poll();
            if (actions.quit)
                break;
            if (actions.pause && !opts.sequence_files.empty())
                playback_paused = !playback_paused;
            if (!opts.sequence_files.empty() && session->done() && exported && !playback_paused &&
                !last_sequence_frame(opts)) {
                select_sequence_frame(opts, opts.sequence_index + 1);
                actions.restart = true;
            }
            if (actions.reference)
                comparison = comparison == 1 ? 0 : 1;
            if (actions.error_view)
                comparison = comparison == 2 ? 0 : 2;
            if (actions.cancel) {
                session->cancel();
                cancelled = true;
            }
            if (actions.pause && !session->done())
                session->set_paused(!session->paused());
            if (actions.denoise) {
                opts.denoise = !opts.denoise;
                opts.neural_active = false;
                notice = "PRESS S TO SAVE";
            }
            if (actions.neural) {
                if (neural_worker) {
                    opts.neural_active = !opts.neural_active;
                    requested_samples = 0;
                    notice = "PRESS S TO SAVE";
                } else
                    notice = "LOAD A MODEL WITH --MODEL";
            }
            if (actions.caustics) {
                opts.render.caustic_photons = opts.render.caustic_photons ? 0 : 300000;
                if (opts.render.caustic_photons)
                    opts.render.transparent_shadows = false;
                actions.restart = true;
            }
            if (actions.glass_shadows) {
                opts.render.transparent_shadows = !opts.render.transparent_shadows;
                if (opts.render.transparent_shadows)
                    opts.render.caustic_photons = 0;
                actions.restart = true;
            }
            if (actions.restart || actions.scene_index >= 0) {
                session->cancel();
                session->wait();
                if (actions.scene_index >= 0) {
                    opts.scene = scene_names().at(actions.scene_index);
                    opts.mesh.clear();
                    opts.scene_file.clear();
                    opts.sequence_files.clear();
                    if (actions.scene_index == 3) {
                        if (!opts.render.caustic_photons)
                            opts.render.caustic_photons = 300000;
                        opts.render.transparent_shadows = false;
                    }
                }
                auto new_scene = build_scene(opts);
                opts.scene = new_scene.name;
                opts.scene_description = scene_description(opts);
                opts.render.history_key = history_description(opts);
                opts.neural_error.clear();
                opts.neural_supported = new_scene.view.defocus_angle == 0 &&
                                        !opts.render.caustic_photons &&
                                        !opts.render.transparent_shadows;
                opts.pipeline_start = std::chrono::steady_clock::now();
                session = std::make_unique<render_session>(std::move(new_scene), opts.render);
                ++generation;
                requested_samples = 0;
                neural_preview.reset();
                save_pending = false;
                frame = session->snapshot();
                preview = frame;
                exported = false;
                cancelled = false;
                completion_handled = false;
                notice.clear();
            }
            double exposure = std::clamp(opts.exposure + actions.exposure_delta, -8.0, 8.0);
            if (exposure != opts.exposure) {
                opts.exposure = exposure;
                notice = "PRESS S TO SAVE";
            }
            auto stats = session->stats();
            if (stats.samples != frame.samples) {
                auto result = session->capture();
                frame = std::move(result.frame);
                stats = result.stats;
            }
            if (session->done() && !completion_handled) {
                session->wait();
                auto result = session->capture();
                frame = std::move(result.frame);
                stats = result.stats;
                completion_handled = true;
                if (!cancelled && !exported) {
                    if (opts.neural_active && neural_worker && opts.neural_supported) {
                        save_pending = true;
                        save_target = frame.samples;
                    } else {
                        try {
                            save(opts, frame, stats);
                            notice = "SAVED IMAGE + JSON";
                        } catch (const std::exception &error) {
                            notice = "SAVE FAILED - SEE TERMINAL";
                            std::cerr << error.what() << '\n';
                        }
                        exported = true;
                        if (opts.quit_after_render && last_sequence_frame(opts))
                            quit = true;
                    }
                }
            }
            if (actions.save) {
                if (opts.neural_active && neural_worker && opts.neural_supported) {
                    save_pending = true;
                    save_target = frame.samples;
                    requested_samples = 0;
                    notice = "RECONSTRUCTING FOR SAVE";
                } else {
                    auto result = session->capture();
                    frame = std::move(result.frame);
                    stats = result.stats;
                    try {
                        auto output = save(opts, frame, stats);
                        notice = "SAVED IMAGE + JSON";
                        std::cout << "Saved " << output << '\n';
                    } catch (const std::exception &error) {
                        notice = "SAVE FAILED - SEE TERMINAL";
                        std::cerr << error.what() << '\n';
                    }
                }
            }
            if (neural_worker) {
                if (auto result = neural_worker->poll();
                    result && result->generation == generation) {
                    opts.neural_seconds = result->seconds;
                    opts.neural_details = result->details;
                    opts.neural_error = result->error;
                    if (!result->error.empty()) {
                        std::cerr << "Neural fallback: " << result->error << '\n';
                        notice = "AI UNAVAILABLE - RAW FALLBACK";
                        opts.neural_active = false;
                        if (save_pending) {
                            result->raw = frame;
                            result->frame = frame;
                            result->stats = stats;
                        }
                    }
                    neural_preview = std::move(result);
                    if (save_pending && neural_preview->raw.samples >= save_target) {
                        try {
                            save(opts, neural_preview->raw, neural_preview->stats,
                                 neural_preview->frame.reconstructed ? &neural_preview->frame
                                                                     : nullptr);
                            notice = "SAVED IMAGE + JSON";
                        } catch (const std::exception &error) {
                            notice = "SAVE FAILED - SEE TERMINAL";
                            std::cerr << error.what() << '\n';
                        }
                        save_pending = false;
                        exported = true;
                        if (opts.quit_after_render && session->done() && last_sequence_frame(opts))
                            quit = true;
                    }
                }
                if (opts.neural_active && opts.neural_supported &&
                    frame.samples > requested_samples &&
                    (requested_samples == 0 ||
                     std::chrono::steady_clock::now() - neural_request_time >=
                         std::chrono::milliseconds(opts.neural_interval_ms) ||
                     session->done() || save_pending)) {
                    neural_worker->submit(frame, stats, generation);
                    requested_samples = frame.samples;
                    neural_request_time = std::chrono::steady_clock::now();
                }
                if (opts.neural_active && !opts.neural_supported)
                    notice = "AI DOMAIN FALLBACK - RAW";
            }
            std::string state = cancelled           ? "CANCELLED"
                                : session->done()   ? "COMPLETE"
                                : session->paused() ? "PAUSED"
                                                    : "RENDERING";
            if (opts.render.caustic_photons)
                state += " / CAUSTICS";
            else if (opts.render.transparent_shadows)
                state += " / GLASS APPROX";
            if (opts.neural_active && opts.neural_supported)
                state += " / AI";
            auto now = std::chrono::steady_clock::now();
            bool needs_preview = frame.samples != preview.samples ||
                                 opts.denoise != preview.denoised || preview.reconstructed;
            bool display_ai = opts.neural_active && neural_preview &&
                              neural_preview->frame.reconstructed && opts.neural_supported;
            if (!display_ai && needs_preview &&
                (actions.denoise || actions.neural || session->done() ||
                 now - preview_time > std::chrono::milliseconds(200))) {
                preview = opts.denoise && !frame.guides.empty() ? denoise(frame) : frame;
                preview_time = now;
            }
            if (display_ai &&
                (!preview.reconstructed || preview.samples != neural_preview->frame.samples))
                preview = neural_preview->frame;
            frame_snapshot comparison_frame;
            const frame_snapshot *shown = &preview;
            if (comparison) {
                if (!reference_error.empty() || reference_identity.empty() ||
                    reference.width != preview.width || reference.height != preview.height ||
                    reference_identity != fingerprint(opts, preview.width, preview.height))
                    notice = "REFERENCE MISSING OR MISMATCHED";
                else {
                    comparison_frame = preview;
                    comparison_frame.comparison_view = comparison;
                    for (std::size_t i = 0; i < preview.linear.size(); ++i)
                        for (int c = 0; c < 3; ++c)
                            comparison_frame.linear[i][c] =
                                comparison == 1
                                    ? reference.linear[i][c]
                                    : 4 * std::abs(preview.linear[i][c] - reference.linear[i][c]);
                    shown = &comparison_frame;
                    state += comparison == 1 ? " / REFERENCE" : " / ERROR 4X";
                }
            }
            window.present(*shown, stats, opts.scene, state, opts.render.samples, opts.exposure,
                           notice);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        session->cancel();
        session->wait();
#endif
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "RayTracer: " << error.what() << '\n';
        return 1;
    }
}
