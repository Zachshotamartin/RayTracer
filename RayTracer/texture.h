#pragma once
#include "color.h"
#include <memory>
class texture {
  public:
    virtual ~texture() = default;
    virtual color value(double u, double v, const point3 &p) const = 0;
};
class solid_color : public texture {
  public:
    explicit solid_color(color c) : color_(c) {}
    color value(double, double, const point3 &) const override { return color_; }

  private:
    color color_;
};
class checker_texture : public texture {
  public:
    checker_texture(double scale, color a, color b) : scale_(scale), a_(a), b_(b) {
        if (!(scale > 0))
            throw std::invalid_argument("Checker scale must be positive");
    }
    color value(double, double, const point3 &p) const override {
        double parity = std::fmod(std::floor(p.x() / scale_) + std::floor(p.y() / scale_) +
                                      std::floor(p.z() / scale_),
                                  2.0);
        return parity == 0 ? a_ : b_;
    }

  private:
    double scale_;
    color a_, b_;
};
class band_texture : public texture {
  public:
    band_texture(color a, color b, double bands = 12) : a_(a), b_(b), bands_(bands) {}
    color value(double u, double v, const point3 &) const override {
        double wave = std::sin(2 * rt_pi * (bands_ * v + 0.3 * std::sin(6 * rt_pi * u)));
        double t = std::clamp(0.5 + wave * 6, 0.0, 1.0);
        return (1 - t) * a_ + t * b_;
    }

  private:
    color a_, b_;
    double bands_;
};
