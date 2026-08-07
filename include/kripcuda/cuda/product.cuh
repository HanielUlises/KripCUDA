#pragma once

#include "kripcuda/cuda/device_kripke.cuh"

namespace kripcuda {

/// How two component models are composed into one Kripke structure.
enum class ProductKind {
    /// Asynchronous composition: exactly one component moves per transition.
    /// This is the standard model of interleaved concurrency, and the state
    /// space it generates is what the explorers are meant to attack.
    Interleaving,

    /// Synchronous composition: both components move on every transition.
    Synchronous,
};

/// Builds the product of two device-resident models entirely on the GPU.
///
/// The state space is S₁ × S₂ indexed as s₁·|S₂| + s₂, the propositions of the
/// right operand are shifted above those of the left, and the resulting CSR
/// satisfies the invariants of the model layer by construction: successor lists
/// come out sorted because the index map is monotone in each component, and the
/// only transition a product state can acquire twice — the pair of self loops
/// under interleaving — is accounted for when the degrees are computed.
///
/// Neither operand nor the result crosses the bus: degrees are counted in one
/// kernel, turned into row offsets by a device-wide exclusive scan, and filled
/// by a second kernel. Throws std::overflow_error if the product exceeds the
/// index types.
[[nodiscard]] DeviceKripke buildProduct(const DeviceKripke& left, const DeviceKripke& right,
                                        ProductKind kind, cudaStream_t stream = nullptr);

/// Composition of `copies` identical components, built by folding the product
/// from the left. The state space grows as |S|^copies, so this is the intended
/// way to generate models large enough to be worth a GPU.
[[nodiscard]] DeviceKripke buildPower(const DeviceKripke& component, unsigned copies,
                                      ProductKind kind, cudaStream_t stream = nullptr);

} // namespace kripcuda
