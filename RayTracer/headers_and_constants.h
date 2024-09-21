#ifndef headers_and_constants_h
#define headers_and_constants_h

#include <SDL2/SDL.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

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


// All Headers

#include "sdl_window.h"
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "color.h"


#endif
