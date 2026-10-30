#pragma once

#include <cstdint>
#include <cstring>

namespace ORB_SLAM3
{
// The app builds with -ffast-math. Inspect IEEE bits so NaN/Inf diagnostics
// remain meaningful even when the compiler assumes finite arithmetic.
inline bool CornerFinite(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

inline int CornerRayAngleBin(float cosine)
{
    if (!CornerFinite(cosine) || cosine < -1.0f || cosine > 1.0f) return 5;
    if (cosine > 0.9999619231f) return 0; // < 0.5 degree
    if (cosine > 0.9998476952f) return 1; // 0.5 .. 1
    if (cosine > 0.9993908270f) return 2; // 1 .. 2
    if (cosine > 0.9961946981f) return 3; // 2 .. 5
    return 4;                          // >= 5 (including nonpositive cosine)
}

inline int CornerRayRejection(float cosine, bool inertial)
{
    if (!CornerFinite(cosine)) return 0;
    if (cosine < -1.0f || cosine > 1.0f) return 1;
    if (cosine <= 0.0f) return 2;
    if (cosine >= (inertial ? 0.9996f : 0.9998f)) return 3;
    return 4; // Other part of the original stereo/ray condition failed.
}

struct CornerTriangulationStatistics
{
    int triangulation_matches = 0, created = 0, parallax_rejected = 0;
    int triangulation_failed = 0, depth1_rejected = 0, depth2_rejected = 0;
    int reproj1_rejected = 0, reproj2_rejected = 0, zero_distance = 0, far_rejected = 0, scale_rejected = 0;
    int angle_bins[6] = {}; // <.5, .5..1, 1..2, 2..5, >=5, invalid
    int ray_rejected[5] = {}; // nonfinite, out of range, nonpositive, low parallax, other

    void Add(const CornerTriangulationStatistics& other)
    {
        triangulation_matches += other.triangulation_matches;
        created += other.created;
        parallax_rejected += other.parallax_rejected;
        triangulation_failed += other.triangulation_failed;
        depth1_rejected += other.depth1_rejected;
        depth2_rejected += other.depth2_rejected;
        reproj1_rejected += other.reproj1_rejected;
        reproj2_rejected += other.reproj2_rejected;
        zero_distance += other.zero_distance;
        far_rejected += other.far_rejected;
        scale_rejected += other.scale_rejected;
        for (int i = 0; i < 6; ++i) angle_bins[i] += other.angle_bins[i];
        for (int i = 0; i < 5; ++i) ray_rejected[i] += other.ray_rejected[i];
    }
};
}
