//
//  sdl.h
//  RayTracer
//
//  Created by Zachary Martin on 9/20/24.
//

#ifndef sdl_h
#define sdl_h

const char *title = "Ray Tracer";
int width = 1280;
int height = 720;
bool running = false;
SDL_Renderer *renderer;
SDL_Window *window;
SDL_Texture *screen;
SDL_Rect screensize;
SDL_Point mouse;
const Uint8 *keystates;
Uint32 mousestate;
SDL_Event event;

void setDrawColor(SDL_Color c) { SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a); }
void drawPoint(SDL_Point p) { SDL_RenderDrawPoint(renderer, p.x, p.y); }
void drawPoint(int x, int y) { SDL_RenderDrawPoint(renderer, x, y); }

void init(int w, int h) {
    width = w;
    height = h;
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        printf("Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
    std::cout << w << " " << h << std::endl;
    window = SDL_CreateWindow(title,
                              SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED,
                              width,
                              height,
                              SDL_WINDOW_SHOWN
                              );
    renderer = SDL_CreateRenderer(window, 
                                  -1,
                                  SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
                                  );
    SDL_SetRenderDrawBlendMode(renderer, 
                               SDL_BLENDMODE_BLEND
                               );
    screen = SDL_CreateTexture(renderer,
                               SDL_PIXELFORMAT_RGBA8888,
                               SDL_TEXTUREACCESS_TARGET,
                               width,
                               height
                               );
    screensize.x=screensize.y=0;
    screensize.w=width;screensize.h=height;
    running = 1;
    
}

void render() {
    SDL_SetRenderTarget(renderer, NULL);
    SDL_RenderCopy(renderer, screen, &screensize, &screensize);
    SDL_RenderPresent(renderer);
    keystates = SDL_GetKeyboardState(NULL);
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT)
        running = false;
    }
    mousestate = SDL_GetMouseState(&mouse.x, &mouse.y);
    SDL_SetRenderTarget(renderer, screen);
}

void quit() {
  running = 0;
  SDL_DestroyRenderer(renderer);
  SDL_DestroyTexture(screen);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

#endif /* sdl_h */
