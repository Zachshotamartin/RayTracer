#ifndef headers_and_constants_h
#define headers_and_constants_h

#include <SDL2/SDL.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <random>
// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}
// Returns the discriminant
inline double discriminant(double a, double b, double c) {
    return (b * b) - a * c;
}
inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

    
// All Headers
#include "interval.h"
#include "sdl_window.h"
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"

#include "sphere.h"

#include "color.h"
#include "material.h"
#include "camera.h"



#endif
