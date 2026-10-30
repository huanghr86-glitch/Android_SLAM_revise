#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ORB_SLAM3
{
// Bit checks remain valid under the Android build's -ffast-math.
inline bool RecoveryFinite(double value)
{
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

inline bool PnPCanSample(int completed, int limit, int batch, int batch_limit)
{
    return completed < limit && batch < batch_limit;
}

inline int PnPRequiredIterations(double probability, double epsilon, int sample_size, int cap)
{
    if (cap < 1) return 0;
    if (epsilon >= 1.0) return 1;
    if (!(epsilon > 0) || !(probability > 0 && probability < 1)) return cap;
    const double success = std::pow(epsilon, sample_size);
    if (!(success > 0)) return cap;
    const double count = std::ceil(std::log1p(-probability) / std::log1p(-success));
    if (!RecoveryFinite(count) || count >= cap) return cap;
    return std::max(1, static_cast<int>(count));
}

// The same transfer is used for minimal and refined poses, before checking
// inliers. A rejected result leaves the previous members untouched.
template<class Matrix>
bool ApplyPnPResult(const Matrix& result, double (&rotation)[3][3], double (&translation)[3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (!RecoveryFinite(result(i, j))) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double dot = 0;
            for (int k = 0; k < 3; ++k) dot += result(i, k) * result(j, k);
            if (std::abs(dot - (i == j ? 1.0 : 0.0)) > .01) return false;
        }
    const double det = result(0,0)*(result(1,1)*result(2,2)-result(1,2)*result(2,1))
                     - result(0,1)*(result(1,0)*result(2,2)-result(1,2)*result(2,0))
                     + result(0,2)*(result(1,0)*result(2,1)-result(1,1)*result(2,0));
    if (std::abs(det - 1.0) > .01) return false;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) rotation[i][j] = result(i, j);
        translation[i] = result(i, 3);
    }
    return true;
}

constexpr int kFallbackMaxStreak = 3;
constexpr double kFallbackMaxFrameGap = .15;
constexpr int kFallbackMinimumUnique = 30;
constexpr int kFallbackMinimumCells = 10;
constexpr double kFallbackMaxRotation = 5.0;
constexpr double kFallbackMaxTranslationDepth = .03;
constexpr double kRecoveryChi2 = 5.991;
constexpr double kShortRecoveryWindow = 1.0;
constexpr double kShortAnchorRotation = 12.0;
constexpr double kShortAnchorTranslationDepth = .10;
constexpr int kRecoveryPointLogLimit = 96;
constexpr int kShortOptimizationBudget = 3;

inline bool AllowRecoveryOptimization(int correspondences, bool short_enabled, int& extra_calls)
{
    if (correspondences >= 50) return true;
    if (!short_enabled || correspondences < 30 || extra_calls >= kShortOptimizationBudget) return false;
    ++extra_calls;
    return true;
}

inline bool LocalFallbackEligible(bool mono, bool previous_ok, bool recent_relocalization,
                                  int initial_inliers, double frame_gap, int streak)
{
    return mono && previous_ok && !recent_relocalization && initial_inliers >= 15 && initial_inliers < 30
        && RecoveryFinite(frame_gap) && frame_gap > 0 && frame_gap <= kFallbackMaxFrameGap
        && streak < kFallbackMaxStreak;
}

struct RecoveryQuality
{
    int associations = 0, unique = 0, duplicates = 0, invalid = 0, cells = 0, extra = 0;
    double rmse = -1, median = -1, p95 = -1, rotation_deg = -1, translation_depth = -1;
    double normalized_rmse = -1, normalized_p95 = -1, max_chi2 = -1;
    int chi2_over = 0;
    double median_depth = -1;
};

inline double RecoveryChi2(double pixel_error, double inverse_variance)
{
    if (!RecoveryFinite(pixel_error) || pixel_error < 0 || !RecoveryFinite(inverse_variance) || inverse_variance <= 0)
        return -1;
    return pixel_error * pixel_error * inverse_variance;
}

inline bool AcceptRecoveryGeometry(const RecoveryQuality& q)
{
    return q.unique >= kFallbackMinimumUnique && q.cells >= kFallbackMinimumCells && q.invalid == 0
        && q.chi2_over == 0 && RecoveryFinite(q.normalized_rmse) && q.normalized_rmse >= 0
        && q.normalized_rmse*q.normalized_rmse <= kRecoveryChi2 + .0001
        && RecoveryFinite(q.max_chi2) && q.max_chi2 >= 0 && q.max_chi2 <= kRecoveryChi2 + .0001;
}

inline bool AcceptLocalFallback(const RecoveryQuality& q)
{
    return AcceptRecoveryGeometry(q)
        && RecoveryFinite(q.rotation_deg) && q.rotation_deg >= 0 && q.rotation_deg <= kFallbackMaxRotation
        && RecoveryFinite(q.translation_depth) && q.translation_depth >= 0
        && q.translation_depth <= kFallbackMaxTranslationDepth;
}

inline bool AcceptShortAnchor(const RecoveryQuality& q, double age)
{
    return AcceptRecoveryGeometry(q) && RecoveryFinite(age) && age > 0 && age <= kShortRecoveryWindow
        && RecoveryFinite(q.rotation_deg) && q.rotation_deg >= 0 && q.rotation_deg <= kShortAnchorRotation
        && RecoveryFinite(q.translation_depth) && q.translation_depth >= 0
        && q.translation_depth <= kShortAnchorTranslationDepth;
}

struct ShortRecoveryChain
{
    bool active = false;
    uint64_t map = 0, frame = 0;
    double timestamp = -1;
    int streak = 0, points = 0;
    void Reset() { *this = ShortRecoveryChain{}; }
    bool Observe(uint64_t map_id, uint64_t frame_id, double time, int count, int shared)
    {
        if (!RecoveryFinite(time) || count < kFallbackMinimumUnique) { Reset(); return false; }
        const bool continues = active && map == map_id && frame_id > frame && time > timestamp
            && time-timestamp <= kFallbackMaxFrameGap && shared >= std::max(15, (points+1)/2);
        streak = continues ? streak+1 : 1;
        active = true; map = map_id; frame = frame_id; timestamp = time; points = count;
        return streak >= 2;
    }
};
}
