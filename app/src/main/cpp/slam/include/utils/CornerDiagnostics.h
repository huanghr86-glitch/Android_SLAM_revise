#pragma once

#include <string>
#include <vector>

namespace ORB_SLAM3
{
// Diagnostic-only helpers. These do not change matching or mapping decisions.
void SetCornerFrameTime(double timestamp);
void SetCornerShortRecoveryEnabled(bool enabled);
bool CornerShortRecoveryEnabled();
double CornerFrameTime();
void ResetCornerPointTotals();
void RecordCornerPointInvalidation(int reason, double birth_timestamp);
void WriteCornerPointTotals(long keyframe, double timestamp);
void WriteCornerEvent(const char* filename, const char* format, ...)
        __attribute__((format(printf, 2, 3)));
void AppendCornerEvent(std::vector<std::string>& lines, const char* format, ...)
        __attribute__((format(printf, 2, 3)));
void WriteCornerEvents(const char* filename, const std::vector<std::string>& lines);

struct ProjectionMatchDiagnostics
{
    int attempted = 0, no_features = 0, no_free_features = 0;
    int descriptor_rejected = 0, ratio_rejected = 0, matched = 0;
};

struct RecoveryProjectionDiagnostics
{
    int tested = 0, seed_excluded = 0, already_found = 0, invalid = 0;
    int depth = 0, bounds = 0, distance = 0, no_features = 0, no_free = 0;
    int descriptor = 0, orientation = 0, matched = 0;
};

struct TriangulationMatchDiagnostics
{
    int shared_nodes = 0, source_slots = 0, source_mapped = 0;
    int source_shared = 0, source_mapped_shared = 0, source_stereo = 0, source_unmatched = 0;
    int source_matches = 0, pairs = 0, pair_used = 0, pair_stereo = 0;
    int pair_descriptor = 0, pair_epipole = 0, pair_epipolar = 0, pair_accepted = 0;
    int orientation = 0, matched = 0, target_unique = 0;
};
}
