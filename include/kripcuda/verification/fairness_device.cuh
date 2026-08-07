#pragma once

#include "kripcuda/verification/scc_device.cuh"

#include <vector>

namespace kripcuda {

/// A generalised Büchi fairness condition: a set of constraints, each a set of
/// states, and a path is fair when it visits every constraint infinitely often.
///
/// Weak and strong fairness of a component are both expressible here once the
/// enabledness and taken-ness of its transitions are exposed as atomic
/// propositions of the model, which is why this layer knows only about the
/// generalised Büchi form.
class FairnessCondition {
public:
    FairnessCondition() = default;

    FairnessCondition& add(DeviceStateSet constraint) {
        constraints_.push_back(std::move(constraint));
        return *this;
    }

    [[nodiscard]] std::size_t size() const noexcept { return constraints_.size(); }
    [[nodiscard]] bool empty() const noexcept { return constraints_.empty(); }
    [[nodiscard]] const DeviceStateSet& operator[](std::size_t index) const {
        return constraints_[index];
    }

private:
    std::vector<DeviceStateSet> constraints_;
};

/// The states of a fair strongly connected component of the subgraph induced by
/// `region`: a component that lies on a cycle and meets every constraint.
///
/// An unconstrained condition makes every nontrivial component fair, so the
/// result degenerates to the cyclic states of the region — which is the right
/// reading, since with no fairness requirement every cycle is fair.
[[nodiscard]] DeviceStateSet fairComponentStates(const DeviceKripke& model,
                                                 const DeviceStateSet& region,
                                                 const FairnessCondition& fairness,
                                                 const ReverseRelation& reverse,
                                                 cudaStream_t stream = nullptr);

/// EG_fair φ: the states from which some path stays inside ⟦φ⟧ for ever and is
/// fair.
///
/// Such a path must enter a fair component of the subgraph induced by φ and
/// stay there, so the set is the backward reachability of the fair components
/// within φ — the least fixpoint μX. F ∪ (φ ∩ pre∃(X)), which is the same step
/// the unfair evaluator already uses for E[φ U ψ].
[[nodiscard]] DeviceStateSet checkFairExistsGlobally(const DeviceKripke& model,
                                                     const DeviceStateSet& condition,
                                                     const FairnessCondition& fairness,
                                                     const ReverseRelation& reverse,
                                                     cudaStream_t stream = nullptr);

struct FairCycleResult {
    /// Whether some initial state starts a fair path.
    bool holds = false;
    /// States from which a fair path exists.
    StateId witnessCount = 0;
    SccStatistics scc;
};

/// Whether the model admits a fair path from an initial state — the emptiness
/// check that language containment reduces to once the property automaton has
/// been composed into the model.
[[nodiscard]] FairCycleResult checkFairCycle(const DeviceKripke& model,
                                             const FairnessCondition& fairness,
                                             cudaStream_t stream = nullptr);

} // namespace kripcuda
