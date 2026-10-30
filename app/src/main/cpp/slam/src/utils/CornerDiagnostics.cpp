#include "utils/CornerDiagnostics.h"
#include <android/log.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <cstdint>

namespace ORB_SLAM3
{
namespace
{
std::atomic<double> latest_frame_time{-1.0};
std::atomic<bool> short_recovery_enabled{true};
std::mutex log_mutex;
// Low 32 bits: all invalidations; high 32 bits: age <= 2 s.
std::atomic<uint64_t> point_totals[5]{};
}

void SetCornerFrameTime(double timestamp) { latest_frame_time.store(timestamp); }
void SetCornerShortRecoveryEnabled(bool enabled) { short_recovery_enabled.store(enabled); }
bool CornerShortRecoveryEnabled() { return short_recovery_enabled.load(); }
double CornerFrameTime() { return latest_frame_time.load(); }

void ResetCornerPointTotals() { for (auto& count : point_totals) count.store(0); }

void RecordCornerPointInvalidation(int reason, double birth_timestamp)
{
    if (reason < 0 || reason >= 5) reason = 0;
    const double age = CornerFrameTime() - birth_timestamp;
    const bool recent = birth_timestamp >= 0 && age >= 0 && age <= 2.0;
    point_totals[reason].fetch_add(1ULL + (recent ? (1ULL << 32) : 0ULL));
}

void WriteCornerPointTotals(long keyframe, double timestamp)
{
    unsigned long long all[5], recent[5];
    for (int i = 0; i < 5; ++i)
    {
        const uint64_t value = point_totals[i].load();
        all[i] = value & 0xffffffffULL;
        recent[i] = value >> 32;
    }
    WriteCornerEvent("SlamMapping.txt", "SLAM_POINT_TOTAL kf=%ld ts=%.9f other=%llu erase_obs=%llu fusion=%llu cull_ratio=%llu cull_age_obs=%llu recent_other=%llu recent_erase_obs=%llu recent_fusion=%llu recent_cull_ratio=%llu recent_cull_age_obs=%llu",
            keyframe, timestamp, all[0], all[1], all[2], all[3], all[4],
            recent[0], recent[1], recent[2], recent[3], recent[4]);
}

void WriteCornerEvent(const char* filename, const char* format, ...)
{
    char line[2048];
    va_list arguments;
    va_start(arguments, format);
    const int length = std::vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    std::lock_guard<std::mutex> guard(log_mutex);
    __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", line);
    char path[192];
    std::snprintf(path, sizeof(path), "/storage/emulated/0/Documents/%s", filename);
    if (FILE* file = std::fopen(path, "a"))
    {
        std::fprintf(file, "%s\n", line);
        std::fclose(file);
    }
    else
        __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_IO file=%s write_failed=1", filename);
    if (length >= static_cast<int>(sizeof(line)))
        __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_IO file=%s truncated=1", filename);
}

void AppendCornerEvent(std::vector<std::string>& lines, const char* format, ...)
{
    char line[2048];
    va_list arguments;
    va_start(arguments, format);
    const int length = std::vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    lines.emplace_back(line);
    if (length >= static_cast<int>(sizeof(line)))
        __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_IO batch_truncated=1");
}

void WriteCornerEvents(const char* filename, const std::vector<std::string>& lines)
{
    if (lines.empty()) return;
    std::lock_guard<std::mutex> guard(log_mutex);
    char path[192];
    std::snprintf(path, sizeof(path), "/storage/emulated/0/Documents/%s", filename);
    FILE* file = std::fopen(path, "a");
    for (const auto& line : lines)
    {
        __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", line.c_str());
        if (file) std::fprintf(file, "%s\n", line.c_str());
    }
    if (file) std::fclose(file);
    else __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_IO file=%s write_failed=1", filename);
}
}
