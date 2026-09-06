#pragma once
#include <cmath>
#include <limits>
inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline constexpr double rt_pi = 3.14159265358979323846;
inline double degrees_to_radians(double degrees) {
    return degrees * rt_pi / 180.0;
}
inline double discriminant(double a, double b, double c) {
    return b * b - a * c;
}
