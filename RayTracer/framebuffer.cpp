#include "framebuffer.h"
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <memory>

frame_snapshot upscale_bilinear(const frame_snapshot &frame, int scale) {
    if ((scale != 1 && scale != 2) || frame.width < 1 || frame.height < 1 ||
        frame.linear.size() != std::size_t(frame.width) * frame.height ||
        std::uint64_t(frame.width) * frame.height * scale * scale > 16777216)
        throw std::invalid_argument("Invalid bilinear output dimensions");
    if (scale == 1)
        return frame;
    frame_snapshot out;
    out.width = frame.width * scale;
    out.height = frame.height * scale;
    out.samples = frame.samples;
    out.camera = frame.camera;
    out.denoised = frame.denoised;
    out.linear.resize(std::size_t(out.width) * out.height);
    for (int y = 0; y < out.height; ++y) {
        const double sy = std::max(0., (y + .5) / scale - .5);
        const int y0 = int(sy), y1 = std::min(y0 + 1, frame.height - 1);
        const double fy = sy - y0;
        for (int x = 0; x < out.width; ++x) {
            const double sx = std::max(0., (x + .5) / scale - .5);
            const int x0 = int(sx), x1 = std::min(x0 + 1, frame.width - 1);
            const double fx = sx - x0;
            auto row = [&](int iy) {
                return frame.linear[std::size_t(iy) * frame.width + x0] * (1 - fx) +
                       frame.linear[std::size_t(iy) * frame.width + x1] * fx;
            };
            out.linear[std::size_t(y) * out.width + x] = row(y0) * (1 - fy) + row(y1) * fy;
        }
    }
    return out;
}
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

std::vector<std::uint8_t> frame_snapshot::rgb(double exposure) const {
    std::vector<std::uint8_t> result(linear.size() * 3);
    const double scale = std::exp2(exposure);
    for (std::size_t i = 0; i < linear.size(); ++i)
        for (int c = 0; c < 3; ++c) {
            double x = std::max(0.0, linear[i][c] * scale);
            // Fitted ACES filmic curve followed by the sRGB transfer function.
            double mapped =
                (std::isfinite(x) && x < 1e6)
                    ? std::clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
                    : 1;
            double srgb =
                mapped <= 0.0031308 ? 12.92 * mapped : 1.055 * std::pow(mapped, 1 / 2.4) - 0.055;
            result[3 * i + c] = static_cast<std::uint8_t>(std::lround(255 * srgb));
        }
    return result;
}
namespace {
using bytes = std::vector<std::uint8_t>;
void be32(bytes &out, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}
std::uint32_t crc32(const std::uint8_t *data, std::size_t size) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            auto c = n;
            for (int k = 0; k < 8; ++k)
                c = (c >> 1) ^ ((c & 1) ? 0xedb88320U : 0);
            values[n] = c;
        }
        return values;
    }();
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i)
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 255];
    return crc ^ 0xffffffffU;
}
void chunk(bytes &png, const char *type, const bytes &data) {
    be32(png, static_cast<std::uint32_t>(data.size()));
    auto start = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());
    be32(png, crc32(png.data() + start, data.size() + 4));
}
} // namespace
void write_png(const std::filesystem::path &path, const frame_snapshot &frame, double exposure) {
    if (frame.width <= 0 || frame.height <= 0 ||
        frame.linear.size() != std::size_t(frame.width) * frame.height)
        throw std::invalid_argument("Invalid framebuffer for PNG output");
    auto pixels = frame.rgb(exposure);
    bytes raw;
    for (int y = 0; y < frame.height; ++y) {
        raw.push_back(0); // PNG filter: none.
        auto begin = pixels.begin() + std::size_t(y) * frame.width * 3;
        raw.insert(raw.end(), begin, begin + frame.width * 3);
    }
    int compressed_size = 0;
    std::unique_ptr<unsigned char, decltype(&std::free)> compressed(
        stbi_zlib_compress(raw.data(), static_cast<int>(raw.size()), &compressed_size, 8),
        std::free);
    if (!compressed)
        throw std::runtime_error("PNG compression failed");
    bytes zlib(compressed.get(), compressed.get() + compressed_size);
    bytes png{137, 80, 78, 71, 13, 10, 26, 10}, header;
    be32(header, frame.width);
    be32(header, frame.height);
    header.insert(header.end(), {8, 2, 0, 0, 0});
    chunk(png, "IHDR", header);
    chunk(png, "sRGB", bytes{0});
    chunk(png, "IDAT", zlib);
    chunk(png, "IEND", {});
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(png.data()),
               static_cast<std::streamsize>(png.size()));
    if (!file)
        throw std::runtime_error("Could not write PNG: " + path.string());
}

namespace {
std::vector<float> linear_pixels(const frame_snapshot &frame, bool rgbe) {
    if (frame.width <= 0 || frame.height <= 0 ||
        frame.linear.size() != std::size_t(frame.width) * frame.height)
        throw std::invalid_argument("Invalid framebuffer for linear output");
    std::vector<float> pixels;
    pixels.reserve(frame.linear.size() * 3);
    for (auto pixel : frame.linear)
        for (int c = 0; c < 3; ++c) {
            double limit = rgbe ? std::ldexp(1.0, 127) : std::numeric_limits<float>::max();
            if (!std::isfinite(pixel[c]) || std::fabs(pixel[c]) >= limit || (rgbe && pixel[c] < 0))
                throw std::invalid_argument("Radiance is outside the export format's range");
            float value = static_cast<float>(pixel[c]);
            if (rgbe && double(value) >= limit)
                throw std::invalid_argument("Rounded radiance exceeds RGBE's range");
            pixels.push_back(value);
        }
    return pixels;
}
std::ofstream output_file(const std::filesystem::path &path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Could not open image: " + path.string());
    return out;
}
} // namespace
void write_hdr(const std::filesystem::path &path, const frame_snapshot &frame) {
    auto pixels = linear_pixels(frame, true);
    auto out = output_file(path);
    auto callback = [](void *context, void *data, int size) {
        static_cast<std::ofstream *>(context)->write(static_cast<char *>(data), size);
    };
    if (!stbi_write_hdr_to_func(callback, &out, frame.width, frame.height, 3, pixels.data()) ||
        !out)
        throw std::runtime_error("Could not write HDR: " + path.string());
}
void write_pfm(const std::filesystem::path &path, const frame_snapshot &frame) {
    auto pixels = linear_pixels(frame, false);
    auto out = output_file(path);
    out << "PF\n" << frame.width << ' ' << frame.height << "\n-1.0\n";
    // PFM scanlines run bottom to top. Emit little-endian IEEE float32 explicitly.
    for (int y = frame.height - 1; y >= 0; --y)
        for (int x = 0; x < frame.width * 3; ++x) {
            auto bits = std::bit_cast<std::uint32_t>(pixels[std::size_t(y) * frame.width * 3 + x]);
            for (int shift = 0; shift < 32; shift += 8)
                out.put(static_cast<char>((bits >> shift) & 255));
        }
    if (!out)
        throw std::runtime_error("Could not write PFM: " + path.string());
}
void write_image(const std::filesystem::path &path, const frame_snapshot &frame, double exposure) {
    auto extension = path.extension().string();
    if (extension == ".png")
        write_png(path, frame, exposure);
    else if (extension == ".hdr")
        write_hdr(path, frame);
    else if (extension == ".pfm")
        write_pfm(path, frame);
    else
        throw std::invalid_argument("Output must end in .png, .hdr, or .pfm");
}

frame_snapshot read_pfm(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    std::string magic;
    int width = 0, height = 0;
    double scale = 0;
    in >> magic >> width >> height >> scale;
    char newline = static_cast<char>(in.get());
    if (newline == '\r')
        newline = static_cast<char>(in.get());
    if (!in || magic != "PF" || width < 1 || height < 1 ||
        std::uint64_t(width) * height > 16777216 || !std::isfinite(scale) || scale == 0 ||
        newline != '\n')
        throw std::invalid_argument("Invalid RGB PFM header");
    frame_snapshot result{width, height, 0, std::vector<color>(std::size_t(width) * height)};
    for (int y = height - 1; y >= 0; --y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
                std::uint32_t bits = 0;
                for (int i = 0; i < 4; ++i) {
                    int byte = in.get();
                    if (byte == EOF)
                        throw std::invalid_argument("Truncated PFM");
                    bits |= std::uint32_t(byte) << (scale < 0 ? 8 * i : 8 * (3 - i));
                }
                double value = std::bit_cast<float>(bits) * std::abs(scale);
                if (!std::isfinite(value))
                    throw std::invalid_argument("Non-finite PFM");
                result.linear[std::size_t(y) * width + x][c] = value;
            }
    if (in.peek() != EOF)
        throw std::invalid_argument("Trailing PFM data");
    return result;
}
