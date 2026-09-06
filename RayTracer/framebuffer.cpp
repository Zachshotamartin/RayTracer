#include "framebuffer.h"
#include <array>
#include <fstream>

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
    // RFC 1950/1951: a zlib stream containing uncompressed DEFLATE blocks.
    // Deliberately dependency-free; output size trades compression for portability.
    bytes zlib{0x78, 0x01};
    for (std::size_t offset = 0; offset < raw.size();) {
        auto length = static_cast<std::uint16_t>(std::min<std::size_t>(65535, raw.size() - offset));
        zlib.push_back(offset + length == raw.size() ? 1 : 0);
        zlib.push_back(length & 255);
        zlib.push_back(length >> 8);
        auto inverse = static_cast<std::uint16_t>(~length);
        zlib.push_back(inverse & 255);
        zlib.push_back(inverse >> 8);
        zlib.insert(zlib.end(), raw.begin() + offset, raw.begin() + offset + length);
        offset += length;
    }
    std::uint32_t a = 1, b = 0;
    for (auto byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    be32(zlib, (b << 16) | a);
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
