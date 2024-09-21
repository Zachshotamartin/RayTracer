#ifndef color_h
#define color_h


// Created to differentiate a positional / ray vec3 from color vector
using color = vec3;

inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

#endif /* color_h */
