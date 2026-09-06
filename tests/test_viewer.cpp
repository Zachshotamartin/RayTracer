#include "sdl_window.h"
#define SDL_MAIN_HANDLED
#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif
int main() {
    try {
        sdl_window viewer(64, 36);
        frame_snapshot frame{64, 36, 1, std::vector<color>(64 * 36, color(0.2, 0.4, 0.6))};
        render_stats stats;
        stats.samples = 1;
        stats.workers = 4;
        viewer.present(frame, stats, "studio", "RENDERING", 4, 0, "");
        SDL_Event event{};
        event.type = SDL_KEYDOWN;
        for (auto key : {SDLK_2, SDLK_SPACE, SDLK_s, SDLK_ESCAPE, SDLK_EQUALS}) {
            event.key.keysym.sym = key;
            if (SDL_PushEvent(&event) < 0)
                throw std::runtime_error(SDL_GetError());
        }
        auto actions = viewer.poll();
        if (actions.scene_index != 1 || !actions.pause || !actions.save || !actions.cancel ||
            actions.exposure_delta != 0.25)
            throw std::runtime_error("Viewer keyboard controls failed");
        frame.width = 32;
        frame.height = 18;
        frame.linear.resize(32 * 18);
        viewer.present(frame, stats, "field", "CANCELLED", 4, 0.25, "SAVED PNG");
        event = {};
        event.type = SDL_QUIT;
        SDL_PushEvent(&event);
        if (!viewer.poll().quit)
            throw std::runtime_error("Viewer close event failed");
        std::cout << "PASS viewer texture updates, resize, controls, and shutdown\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
