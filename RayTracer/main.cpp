#include "denoiser.h"
#include "renderer.h"
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
    if (opts.mesh.empty())
        return make_scene(opts.scene, opts.render.seed);
    std::string warnings;
    auto result = make_mesh_scene(opts.mesh, warnings);
    if (!warnings.empty())
        std::cerr << "OBJ: " << warnings;
    return result;
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
    std::cout << "RayTracer - progressive CPU path tracer\n"
                 "Usage: raytracer [--headless] [options]\n"
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
                 "Viewer: 1/2/3/4 scenes, D denoise, C caustics, G glass shadows, Space pause, R "
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
        << ",\n  \"bvh_build_seconds\": " << stats.build_seconds
        << ",\n  \"path_rays\": " << stats.path_rays
        << ",\n  \"shadow_rays\": " << stats.shadow_rays << ",\n  \"exposure\": " << opts.exposure
        << ",\n  \"mesh_path\": " << quote(opts.mesh.generic_string())
        << ",\n  \"denoised\": " << (opts.denoise ? "true" : "false")
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
std::filesystem::path save(options &opts, const frame_snapshot &frame, const render_stats &stats) {
    auto output =
        opts.output.empty() ? opts.default_output_directory / (opts.scene + ".png") : opts.output;
    if (frame.samples == 0)
        throw std::runtime_error("No complete sample pass is available to save");
    auto start = std::chrono::steady_clock::now();
    frame_snapshot filtered;
    if (opts.denoise)
        filtered = denoise(frame);
    auto processed = std::chrono::steady_clock::now();
    opts.denoise_seconds =
        opts.denoise ? std::chrono::duration<double>(processed - start).count() : 0;
    write_image(output, opts.denoise ? filtered : frame, opts.exposure);
    opts.image_write_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - processed).count();
    auto metadata = output;
    metadata.replace_extension(".json");
    std::ofstream json(metadata);
    json << report(opts, frame, stats);
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
        if (!opts.output.empty() && opts.output.extension() != ".png" &&
            opts.output.extension() != ".hdr" && opts.output.extension() != ".pfm")
            throw std::invalid_argument("Output must end in .png, .hdr, or .pfm");
        if (!opts.mesh.empty())
            opts.scene = "mesh";
        opts.render.collect_guides = opts.denoise || !opts.headless;
        auto content = build_scene(opts);
        opts.render.validate(content.view.aspect_ratio);
        std::signal(SIGINT, interrupt_handler);
        std::signal(SIGTERM, interrupt_handler);
        if (opts.headless) {
            render_session session(std::move(content), opts.render);
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
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            session.wait();
            auto frame = session.snapshot();
            auto stats = session.stats();
            if (frame.samples > 0)
                save(opts, frame, stats);
            if (!opts.quiet)
                std::cerr << "\r" << opts.scene << ": " << frame.samples << " spp in "
                          << stats.seconds << " s\n";
            std::cout << report(opts, frame, stats);
            return interrupted ? 130 : 0;
        }
#if RT_HAS_SDL
        sdl_window window(opts.render.width, opts.render.height(content.view.aspect_ratio));
        auto working_directory = std::filesystem::current_path();
        if (working_directory == working_directory.root_path())
            opts.default_output_directory = window.preferred_output_directory();
        auto session = std::make_unique<render_session>(std::move(content), opts.render);
        frame_snapshot frame = session->snapshot();
        frame_snapshot preview = frame;
        auto preview_time = std::chrono::steady_clock::now();
        std::string notice;
        bool quit = false, exported = false, cancelled = false, completion_handled = false;
        while (!quit && !interrupted) {
            auto actions = window.poll();
            if (actions.quit)
                break;
            if (actions.cancel) {
                session->cancel();
                cancelled = true;
            }
            if (actions.pause && !session->done())
                session->set_paused(!session->paused());
            if (actions.denoise) {
                opts.denoise = !opts.denoise;
                notice = "PRESS S TO SAVE";
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
                    if (actions.scene_index == 3) {
                        if (!opts.render.caustic_photons)
                            opts.render.caustic_photons = 300000;
                        opts.render.transparent_shadows = false;
                    }
                }
                session = std::make_unique<render_session>(build_scene(opts), opts.render);
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
                    try {
                        save(opts, frame, stats);
                        notice = "SAVED IMAGE + JSON";
                    } catch (const std::exception &error) {
                        notice = "SAVE FAILED - SEE TERMINAL";
                        std::cerr << error.what() << '\n';
                    }
                    exported = true;
                    if (opts.quit_after_render)
                        quit = true;
                }
            }
            if (actions.save) {
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
            std::string state = cancelled           ? "CANCELLED"
                                : session->done()   ? "COMPLETE"
                                : session->paused() ? "PAUSED"
                                                    : "RENDERING";
            if (opts.render.caustic_photons)
                state += " / CAUSTICS";
            else if (opts.render.transparent_shadows)
                state += " / GLASS APPROX";
            auto now = std::chrono::steady_clock::now();
            bool needs_preview =
                frame.samples != preview.samples || opts.denoise != preview.denoised;
            if (needs_preview && (actions.denoise || session->done() ||
                                  now - preview_time > std::chrono::milliseconds(200))) {
                preview = opts.denoise && !frame.guides.empty() ? denoise(frame) : frame;
                preview_time = now;
            }
            window.present(preview, stats, opts.scene, state, opts.render.samples, opts.exposure,
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
