#pragma once

#include <unordered_map>
#include <vector>

namespace ORB_SLAM3
{
template<class Key>
struct RelocalizationScoreGroup
{
    Key best;
    float best_score;
    float accumulated_score;
    int neighbors_used = 0;
    int neighbors_unscored = 0;
};

// The map must contain only scores actually computed for the current query.
// Never consult a keyframe's persistent mRelocScore when accumulating neighbors.
template<class Key>
RelocalizationScoreGroup<Key> AccumulateRelocalizationScores(
        Key seed, float seed_score, const std::vector<Key>& neighbors,
        const std::unordered_map<Key, float>& current_scores)
{
    RelocalizationScoreGroup<Key> result{seed, seed_score, seed_score};
    for (Key neighbor : neighbors)
    {
        const auto score = current_scores.find(neighbor);
        if (score == current_scores.end())
        {
            ++result.neighbors_unscored;
            continue;
        }
        ++result.neighbors_used;
        result.accumulated_score += score->second;
        if (score->second > result.best_score)
        {
            result.best = neighbor;
            result.best_score = score->second;
        }
    }
    return result;
}
}
