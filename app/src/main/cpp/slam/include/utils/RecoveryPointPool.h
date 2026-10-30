#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ORB_SLAM3
{
constexpr std::size_t kRecoveryPoolMaxPoints = 2000;
constexpr int kRecoveryPoolRecentKeyFrames = 5;
constexpr int kRecoveryPoolCovisKeyFrames = 10;
constexpr double kRecoveryPoolRecentSeconds = 5.0;
constexpr int kRecoveryPoolMaxExpansions = 3;
constexpr int kRecoveryPoolCandidateLimit = 8;

// IDs, rather than pointer addresses or source slots, define unique map points.
// Track IDs rejected at capacity too, so overflow does not count the same point
// repeatedly when it is observed in multiple keyframes.
template<class Entry>
class RecoveryPointPoolBuilder
{
public:
    explicit RecoveryPointPoolBuilder(std::size_t capacity) : capacity_(capacity)
    {
        points_.reserve(capacity);
        seen_.reserve(capacity * 2);
    }

    bool Add(std::uint64_t point_id, const Entry& entry)
    {
        ++offered;
        if (!seen_.insert(point_id).second) { ++duplicates; return false; }
        if (points_.size() >= capacity_) { ++dropped; return false; }
        points_.push_back(entry);
        return true;
    }

    std::size_t size() const { return points_.size(); }
    std::vector<Entry> Take() { return std::move(points_); }
    std::size_t offered = 0, duplicates = 0, dropped = 0;

private:
    std::size_t capacity_;
    std::unordered_set<std::uint64_t> seen_;
    std::vector<Entry> points_;
};
}
