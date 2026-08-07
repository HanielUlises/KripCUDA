#pragma once

#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/cuda/device_state_set.hpp"
#include "kripcuda/verification/ctl.hpp"

#include <unordered_map>

namespace kripcuda {

/// CTL evaluator over a device-resident model.
///
/// The evaluator owns one device state set per distinct subformula and
/// memoises on node identity, so a subformula shared by several parts of a
/// specification — as the rewritten universal operators produce — is evaluated
/// once. Keeping the evaluator alive across several formulas over the same
/// model reuses both the model and the memoised subresults.
class DeviceCtlEvaluator {
public:
    explicit DeviceCtlEvaluator(const DeviceKripke& model, cudaStream_t stream = nullptr);

    DeviceCtlEvaluator(const DeviceCtlEvaluator&) = delete;
    DeviceCtlEvaluator& operator=(const DeviceCtlEvaluator&) = delete;

    /// Evaluates the formula and returns the satisfying set together with the
    /// verdict for the model's initial states.
    [[nodiscard]] CtlResult check(const ctl::Formula& formula);

    /// The satisfying set left on the device, for composing further device-side
    /// work without a round trip.
    [[nodiscard]] const DeviceStateSet& evaluate(const ctl::Formula& formula);

    [[nodiscard]] std::uint64_t iterations() const noexcept { return iterations_; }
    void resetIterationCount() noexcept { iterations_ = 0; }

private:
    DeviceStateSet& evaluateInto(const ctl::Formula& formula);
    DeviceStateSet makeSet();

    /// next = base ∪ (mask ∩ pre∃(current)), the single step shared by both
    /// temporal fixpoints; a null operand denotes the neutral element.
    /// Returns whether the step changed the set.
    bool fixpointStep(const DeviceStateSet* base, const DeviceStateSet* mask,
                      const DeviceStateSet& current, DeviceStateSet& next);

    const DeviceKripke& model_;
    cudaStream_t stream_;
    std::unordered_map<const void*, DeviceStateSet> memo_;
    DeviceBuffer<std::uint32_t> change_flag_;
    std::uint64_t iterations_ = 0;
};

/// Convenience entry point for a model that is already on the device.
[[nodiscard]] CtlResult checkCtlDevice(const DeviceKripke& model, const ctl::Formula& formula,
                                       cudaStream_t stream = nullptr);

} // namespace kripcuda
