#pragma once

#include <cstdint>
#include <limits>

namespace kripcuda {

using StateId = std::uint32_t;
using PropositionId = std::uint32_t;
using EdgeIndex = std::uint32_t;

/// Labels are stored as a packed bitset; one word covers 64 atomic propositions.
using LabelWord = std::uint64_t;

inline constexpr int kLabelWordBits = 64;
inline constexpr StateId kInvalidState = std::numeric_limits<StateId>::max();
inline constexpr std::int32_t kUnreachable = -1;

constexpr std::uint32_t labelWordsFor(std::uint32_t propositionCount) noexcept {
    return (propositionCount + kLabelWordBits - 1) / kLabelWordBits;
}

} // namespace kripcuda
