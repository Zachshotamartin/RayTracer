#include "sdl_window.h"
#define SDL_MAIN_HANDLED
#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace {
// Small, original 5x7 bitmap glyphs keep the viewer independent of a font runtime.
std::array<unsigned char, 7> glyph(char c) {
    switch (c) {
    case 'A':
        return {14, 17, 17, 31, 17, 17, 17};
    case 'B':
        return {30, 17, 17, 30, 17, 17, 30};
    case 'C':
        return {14, 17, 16, 16, 16, 17, 14};
    case 'D':
        return {30, 17, 17, 17, 17, 17, 30};
    case 'E':
        return {31, 16, 16, 30, 16, 16, 31};
    case 'F':
        return {31, 16, 16, 30, 16, 16, 16};
    case 'G':
        return {14, 17, 16, 23, 17, 17, 15};
    case 'H':
        return {17, 17, 17, 31, 17, 17, 17};
    case 'I':
        return {14, 4, 4, 4, 4, 4, 14};
    case 'J':
        return {7, 2, 2, 2, 18, 18, 12};
    case 'K':
        return {17, 18, 20, 24, 20, 18, 17};
    case 'L':
        return {16, 16, 16, 16, 16, 16, 31};
    case 'M':
        return {17, 27, 21, 21, 17, 17, 17};
    case 'N':
        return {17, 25, 21, 19, 17, 17, 17};
    case 'O':
        return {14, 17, 17, 17, 17, 17, 14};
    case 'P':
        return {30, 17, 17, 30, 16, 16, 16};
    case 'Q':
        return {14, 17, 17, 17, 21, 18, 13};
    case 'R':
        return {30, 17, 17, 30, 20, 18, 17};
    case 'S':
        return {15, 16, 16, 14, 1, 1, 30};
    case 'T':
        return {31, 4, 4, 4, 4, 4, 4};
    case 'U':
        return {17, 17, 17, 17, 17, 17, 14};
    case 'V':
        return {17, 17, 17, 17, 17, 10, 4};
    case 'W':
        return {17, 17, 17, 21, 21, 21, 10};
    case 'X':
        return {17, 17, 10, 4, 10, 17, 17};
    case 'Y':
        return {17, 17, 10, 4, 4, 4, 4};
    case 'Z':
        return {31, 1, 2, 4, 8, 16, 31};
    case '0':
        return {14, 17, 19, 21, 25, 17, 14};
    case '1':
        return {4, 12, 4, 4, 4, 4, 14};
    case '2':
        return {14, 17, 1, 2, 4, 8, 31};
    case '3':
        return {30, 1, 1, 14, 1, 1, 30};
    case '4':
        return {2, 6, 10, 18, 31, 2, 2};
    case '5':
        return {31, 16, 16, 30, 1, 1, 30};
    case '6':
        return {14, 16, 16, 30, 17, 17, 14};
    case '7':
        return {31, 1, 2, 4, 8, 8, 8};
    case '8':
        return {14, 17, 17, 14, 17, 17, 14};
    case '9':
        return {14, 17, 17, 15, 1, 1, 14};
    case '/':
        return {1, 1, 2, 4, 8, 16, 16};
    case '-':
        return {0, 0, 0, 31, 0, 0, 0};
    case '+':
        return {0, 4, 4, 31, 4, 4, 0};
    case '.':
        return {0, 0, 0, 0, 0, 6, 6};
    case ':':
        return {0, 6, 6, 0, 6, 6, 0};
    case '|':
        return {4, 4, 4, 4, 4, 4, 4};
    default:
        return {};
    }
}
void text(SDL_Renderer *renderer, int x, int y, const std::string &label, int scale = 1) {
    for (char ch : label) {
        auto bits = glyph(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col)
                if (bits[row] & (1 << (4 - col))) {
                    SDL_Rect dot{x + col * scale, y + row * scale, scale, scale};
                    SDL_RenderFillRect(renderer, &dot);
                }
        x += 6 * scale;
    }
}
void check(bool condition, const char *action) {
    if (!condition)
        throw std::runtime_error(std::string(action) + ": " + SDL_GetError());
}
} // namespace
struct sdl_window::impl {
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_Texture *texture = nullptr;
    int texture_width = 0, texture_height = 0, last_samples = -1;
    double last_exposure = infinity;
    std::string last_scene, last_title;
    bool last_denoised = false;
    ~impl() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
sdl_window::sdl_window(int width, int height) : impl_(std::make_unique<impl>()) {
    SDL_SetMainReady();
    check(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) == 0, "Initialize SDL");
    int display_width = std::clamp(width, 800, 1280);
    int display_height = display_width * height / width + 124;
    impl_->window =
        SDL_CreateWindow("RayTracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, display_width,
                         display_height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    check(impl_->window, "Create window");
    SDL_SetWindowMinimumSize(impl_->window, 640, 360);
    impl_->renderer =
        SDL_CreateRenderer(impl_->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!impl_->renderer)
        impl_->renderer = SDL_CreateRenderer(impl_->window, -1, SDL_RENDERER_SOFTWARE);
    check(impl_->renderer, "Create renderer");
}
sdl_window::~sdl_window() = default;
std::filesystem::path sdl_window::preferred_output_directory() const {
    std::unique_ptr<char, decltype(&SDL_free)> path(SDL_GetPrefPath("", "RayTracer"), SDL_free);
    check(path != nullptr, "Locate writable export directory");
    std::string text = path.get();
    return std::filesystem::path(std::u8string(text.begin(), text.end())) / "renders";
}
window_actions sdl_window::poll() {
    window_actions result;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
            result.quit = true;
        if (event.type != SDL_KEYDOWN || event.key.repeat)
            continue;
        switch (event.key.keysym.sym) {
        case SDLK_q:
            result.quit = true;
            break;
        case SDLK_ESCAPE:
            result.cancel = true;
            break;
        case SDLK_SPACE:
            result.pause = true;
            break;
        case SDLK_r:
            result.restart = true;
            break;
        case SDLK_s:
            result.save = true;
            break;
        case SDLK_d:
            result.denoise = true;
            break;
        case SDLK_c:
            result.caustics = true;
            break;
        case SDLK_g:
            result.glass_shadows = true;
            break;
        case SDLK_1:
            result.scene_index = 0;
            break;
        case SDLK_2:
            result.scene_index = 1;
            break;
        case SDLK_3:
            result.scene_index = 2;
            break;
        case SDLK_4:
            result.scene_index = 3;
            break;
        case SDLK_EQUALS:
        case SDLK_PLUS:
        case SDLK_KP_PLUS:
            result.exposure_delta += 0.25;
            break;
        case SDLK_MINUS:
        case SDLK_KP_MINUS:
            result.exposure_delta -= 0.25;
            break;
        default:
            break;
        }
    }
    return result;
}
void sdl_window::present(const frame_snapshot &frame, const render_stats &stats,
                         const std::string &scene, const std::string &status, int target,
                         double exposure, const std::string &notice) {
    auto &s = *impl_;
    if (SDL_GetWindowFlags(s.window) & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED))
        return;
    bool resized = frame.width != s.texture_width || frame.height != s.texture_height;
    if (resized) {
        SDL_DestroyTexture(s.texture);
        s.texture = nullptr;
        s.texture = SDL_CreateTexture(s.renderer, SDL_PIXELFORMAT_RGB24,
                                      SDL_TEXTUREACCESS_STREAMING, frame.width, frame.height);
        check(s.texture, "Create preview texture");
        s.texture_width = frame.width;
        s.texture_height = frame.height;
        SDL_SetTextureScaleMode(s.texture, SDL_ScaleModeLinear);
    }
    if (resized || frame.samples != s.last_samples || exposure != s.last_exposure ||
        scene != s.last_scene || frame.denoised != s.last_denoised) {
        auto pixels = frame.rgb(exposure);
        void *target_pixels = nullptr;
        int pitch = 0;
        check(SDL_LockTexture(s.texture, nullptr, &target_pixels, &pitch) == 0,
              "Lock preview texture");
        for (int y = 0; y < frame.height; ++y)
            std::memcpy(static_cast<unsigned char *>(target_pixels) + y * pitch,
                        pixels.data() + std::size_t(y) * frame.width * 3, frame.width * 3);
        SDL_UnlockTexture(s.texture);
        s.last_samples = frame.samples;
        s.last_exposure = exposure;
        s.last_scene = scene;
        s.last_denoised = frame.denoised;
    }
    int width, height;
    SDL_GetWindowSize(s.window, &width, &height);
    SDL_RenderSetLogicalSize(s.renderer, width, height);
    SDL_SetRenderDrawColor(s.renderer, 15, 19, 20, 255);
    SDL_RenderClear(s.renderer);
    int available = std::max(1, height - 124);
    double scale = std::min(double(width) / frame.width, double(available) / frame.height);
    SDL_Rect image{(width - int(frame.width * scale)) / 2,
                   (available - int(frame.height * scale)) / 2, int(frame.width * scale),
                   int(frame.height * scale)};
    check(SDL_RenderCopy(s.renderer, s.texture, nullptr, &image) == 0, "Display preview");
    SDL_SetRenderDrawColor(s.renderer, 213, 191, 129, 255);
    SDL_Rect progress{0, available, width * frame.samples / std::max(1, target), 3};
    SDL_RenderFillRect(s.renderer, &progress);
    std::ostringstream line;
    line << scene << " / " << status << "   " << frame.samples << "/" << target << " SPP   "
         << std::fixed << std::setprecision(1) << stats.seconds << " S   " << stats.workers
         << " CPU";
    int status_scale = line.str().size() * 12 <= static_cast<std::size_t>(width - 32) ? 2 : 1;
    text(s.renderer, 16, available + 15, line.str(), status_scale);
    SDL_SetRenderDrawColor(s.renderer, 162, 174, 172, 255);
    text(s.renderer, 16, available + 40, "1 DEMO  2 FIELD  3 STUDIO  4 CAUSTICS | SPACE PAUSE", 2);
    text(s.renderer, 16, available + 58, "D DENOISE | R RESTART | S SAVE | ESC CANCEL | Q QUIT", 2);
    text(s.renderer, 16, available + 76, "C CAUSTICS | G GLASS SHADOWS | +/- EXPOSURE", 2);
    SDL_SetRenderDrawColor(s.renderer, 213, 191, 129, 255);
    std::ostringstream footer;
    footer << "EXPOSURE " << std::showpos << std::fixed << std::setprecision(2) << exposure << "   "
           << (frame.denoised ? "FILTERED  " : "RAW  ") << notice;
    text(s.renderer, 16, available + 100, footer.str(), 2);
    // Keep the native title stable during each state. Rapid title changes make
    // accessibility snapshots unstable; continuously changing numbers live in the HUD.
    auto title = "RayTracer - " + scene + " / " + status;
    if (title != s.last_title) {
        SDL_SetWindowTitle(s.window, title.c_str());
        s.last_title = title;
    }
    SDL_RenderPresent(s.renderer);
}
