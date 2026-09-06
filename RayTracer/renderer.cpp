#include "renderer.h"
#include "bvh.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <latch>
#include <mutex>
#include <thread>

int render_settings::height(double aspect) const {
    if (!std::isfinite(aspect) || aspect <= 0)
        throw std::invalid_argument("Aspect ratio must be positive and finite");
    double h = width / aspect;
    if (h > 16384)
        throw std::invalid_argument("Image height exceeds 16384");
    return std::max(1, static_cast<int>(h));
}
void render_settings::validate(double aspect) const {
    if (width < 1 || width > 8192 || samples < 1 || samples > 100000 || max_depth < 1 ||
        max_depth > 128 || threads < 0 || threads > 256)
        throw std::invalid_argument(
            "Limits: width 1..8192, samples 1..100000, depth 1..128, threads 0..256");
    if (std::uint64_t(width) * height(aspect) > 16777216)
        throw std::invalid_argument("Image exceeds 16 megapixels");
}
struct render_session::impl {
    using clock = std::chrono::steady_clock;
    scene content;
    render_settings settings;
    std::unique_ptr<camera> view;
    std::shared_ptr<hittable> world;
    int width, height, tile_columns, tile_count, worker_count;
    std::vector<color> accumulation;
    std::atomic<std::uint64_t> path_rays{0}, shadow_rays{0};
    mutable std::mutex frame_mutex, error_mutex, pause_mutex;
    std::mutex phase_mutex;
    std::condition_variable pause_cv, phase_cv;
    int arrivals = 0, phase_generation = 0;
    frame_snapshot published;
    render_stats statistics;
    std::atomic<int> next_tile{0};
    std::atomic<bool> cancelled{false}, is_paused{false}, finished{false};
    bool stop_after_pass = false, launch_failed = false;
    std::exception_ptr error;
    std::latch launch{1};
    std::vector<std::thread> workers;
    clock::time_point start;

    impl(scene s, render_settings opts) : content(std::move(s)), settings(opts) {
        settings.validate(content.view.aspect_ratio);
        width = settings.width;
        height = settings.height(content.view.aspect_ratio);
        view = std::make_unique<camera>(content.view, width, height);
        tile_columns = (width + 15) / 16;
        tile_count = tile_columns * ((height + 15) / 16);
        worker_count = std::min(
            tile_count, settings.threads
                            ? settings.threads
                            : static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        accumulation.resize(std::size_t(width) * height);
        published = {width, height, 0, std::vector<color>(accumulation.size())};
        auto build_start = clock::now();
        if (settings.use_bvh && !content.objects.objects.empty())
            world = std::make_shared<bvh_node>(content.objects.objects);
        else
            world = std::make_shared<hittable_list>(content.objects);
        statistics.build_seconds =
            std::chrono::duration<double>(clock::now() - build_start).count();
        statistics.workers = worker_count;
        statistics.objects = static_cast<int>(content.objects.objects.size());
        start = clock::now();
        try {
            for (int i = 0; i < worker_count; ++i)
                workers.emplace_back([this] { work(); });
        } catch (...) {
            launch_failed = true;
            launch.count_down();
            for (auto &worker : workers)
                worker.join();
            throw;
        }
        launch.count_down();
    }
    void cancel() {
        {
            std::lock_guard lock(pause_mutex);
            cancelled.store(true);
        }
        pause_cv.notify_all();
    }
    void remember_error() {
        {
            std::lock_guard lock(error_mutex);
            if (!error)
                error = std::current_exception();
        }
        cancel();
    }
    void complete_pass() noexcept {
        try {
            std::lock_guard lock(frame_mutex);
            if (!cancelled.load()) {
                ++published.samples;
                double scale = 1.0 / published.samples;
                for (std::size_t i = 0; i < accumulation.size(); ++i)
                    published.linear[i] = accumulation[i] * scale;
                statistics.samples = published.samples;
            }
            statistics.seconds = std::chrono::duration<double>(clock::now() - start).count();
            statistics.path_rays = path_rays.load(std::memory_order_relaxed);
            statistics.shadow_rays = shadow_rays.load(std::memory_order_relaxed);
            stop_after_pass = cancelled.load() || published.samples >= settings.samples;
            next_tile.store(0);
        } catch (...) {
            remember_error();
            stop_after_pass = true;
        }
        if (stop_after_pass)
            finished.store(true, std::memory_order_release);
    }
    void work() {
        launch.wait();
        if (launch_failed)
            return;
        for (int pass = 0;; ++pass) {
            ray_counts counts;
            try {
                while (!cancelled.load(std::memory_order_relaxed)) {
                    if (is_paused.load()) {
                        std::unique_lock lock(pause_mutex);
                        pause_cv.wait(lock,
                                      [this] { return !is_paused.load() || cancelled.load(); });
                    }
                    if (cancelled.load())
                        break;
                    int tile = next_tile.fetch_add(1);
                    if (tile >= tile_count)
                        break;
                    int x0 = (tile % tile_columns) * 16, y0 = (tile / tile_columns) * 16;
                    for (int y = y0; y < std::min(y0 + 16, height); ++y)
                        for (int x = x0; x < std::min(x0 + 16, width); ++x) {
                            if (cancelled.load(std::memory_order_relaxed))
                                break;
                            auto index = std::size_t(y) * width + x;
                            auto rng = sampler::for_pixel(settings.seed, index, pass);
                            auto r = view->get_ray(x, y, rng);
                            accumulation[index] +=
                                trace_path(r, *world, content.lights, content.env,
                                           settings.max_depth, rng, counts);
                        }
                }
            } catch (...) {
                remember_error();
            }
            path_rays.fetch_add(counts.paths, std::memory_order_relaxed);
            shadow_rays.fetch_add(counts.shadows, std::memory_order_relaxed);
            if (finish_pass())
                break;
        }
    }
    bool finish_pass() {
        // A mutex-backed reusable barrier gives both buffers and stop decisions
        // an explicit synchronization edge, including cancellation-only passes.
        std::unique_lock lock(phase_mutex);
        const int generation = phase_generation;
        if (++arrivals == worker_count) {
            complete_pass();
            arrivals = 0;
            ++phase_generation;
            phase_cv.notify_all();
        } else {
            phase_cv.wait(lock, [&] { return phase_generation != generation; });
        }
        // All participants leave with the same decision before another phase
        // can finish. External cancellation must never bypass this barrier.
        return stop_after_pass;
    }
    void join() {
        for (auto &worker : workers)
            if (worker.joinable())
                worker.join();
    }
    ~impl() {
        cancel();
        join();
    }
};
render_session::render_session(scene s, render_settings opts)
    : impl_(std::make_unique<impl>(std::move(s), opts)) {}
render_session::~render_session() = default;
void render_session::cancel() {
    impl_->cancel();
}
void render_session::set_paused(bool value) {
    {
        std::lock_guard lock(impl_->pause_mutex);
        impl_->is_paused.store(value);
    }
    impl_->pause_cv.notify_all();
}
bool render_session::paused() const {
    return impl_->is_paused.load();
}
bool render_session::done() const {
    return impl_->finished.load(std::memory_order_acquire);
}
void render_session::wait() {
    impl_->join();
    std::lock_guard lock(impl_->error_mutex);
    if (impl_->error)
        std::rethrow_exception(impl_->error);
}
frame_snapshot render_session::snapshot() const {
    std::lock_guard lock(impl_->frame_mutex);
    return impl_->published;
}
render_stats render_session::stats() const {
    std::lock_guard lock(impl_->frame_mutex);
    auto result = impl_->statistics;
    if (!impl_->finished.load())
        result.seconds = std::chrono::duration<double>(impl::clock::now() - impl_->start).count();
    return result;
}
render_result render_session::capture() const {
    std::lock_guard lock(impl_->frame_mutex);
    auto stats = impl_->statistics;
    if (!impl_->finished.load())
        stats.seconds = std::chrono::duration<double>(impl::clock::now() - impl_->start).count();
    return {impl_->published, stats};
}
