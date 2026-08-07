#pragma once

#include "kripcuda/kripke.hpp"
#include "kripcuda/state_set.hpp"
#include "kripcuda/verification/formula.hpp"

namespace kripcuda {

/// Result of evaluating a CTL state formula over a model.
struct CtlResult {
    /// The states satisfying the formula: ⟦φ⟧ = { s ∈ S | M, s ⊨ φ }.
    StateSet satisfying;

    /// Whether M ⊨ φ, that is, whether every initial state satisfies it.
    bool holdsInAllInitialStates = false;

    /// Number of fixpoint iterations performed, summed over all subformulas.
    /// Reported because it is the sequential depth of the computation and thus
    /// the quantity that bounds what parallelism can buy.
    std::uint64_t iterations = 0;
};

/// Sequential reference evaluator, used as an oracle for the device path and as
/// a fallback when no CUDA device is available.
[[nodiscard]] CtlResult checkCtlHost(const KripkeStructure& structure, const ctl::Formula& formula);

/// Evaluates a CTL formula on the GPU: one kernel per propositional operator,
/// one fixpoint loop of pre-image kernels per temporal operator.
///
/// Throws kripcuda::CudaError if the CUDA runtime reports a failure.
[[nodiscard]] CtlResult checkCtlDevice(const KripkeStructure& structure,
                                       const ctl::Formula& formula);

} // namespace kripcuda
