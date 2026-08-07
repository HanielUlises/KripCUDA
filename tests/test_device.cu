#include "kripcuda/cuda/device_kripke.cuh"
#include "kripcuda/cuda/device_state_set.cuh"
#include "kripcuda/cuda/product.cuh"
#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/cuda/scan.cuh"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/exploration/reachability.hpp"
#include "kripcuda/exploration/reachability_device.cuh"
#include "kripcuda/verification/ctl.hpp"
#include "kripcuda/verification/ctl_device.cuh"

#include "models.hpp"
#include "test_support.hpp"

#include <iostream>
#include <numeric>
#include <random>

using namespace kripcuda;

namespace {

void testScan() {
    Stream stream;
    DeviceScan scan;

    // Sizes around the tile boundary (256 threads x 4 items) and beyond a
    // single scan block, where the tile-offset pass starts to matter.
    for (const std::size_t count : {std::size_t{1}, std::size_t{1023}, std::size_t{1024},
                                    std::size_t{1025}, std::size_t{300000}}) {
        std::mt19937_64 engine(count);
        std::uniform_int_distribution<std::uint32_t> value(0, 7);

        std::vector<std::uint32_t> host(count);
        for (std::uint32_t& element : host) {
            element = value(engine);
        }

        DeviceBuffer<std::uint32_t> input{std::span<const std::uint32_t>(host)};
        DeviceBuffer<std::uint32_t> output(count);
        const std::uint64_t total = scan.exclusiveScan(input.data(), output.data(), count, stream);

        std::vector<std::uint32_t> expected(count);
        std::exclusive_scan(host.begin(), host.end(), expected.begin(), std::uint32_t{0});
        const std::vector<std::uint32_t> actual = output.toHost(stream);

        EXPECT_EQ(actual == expected, true);
        EXPECT_EQ(total, std::accumulate(host.begin(), host.end(), std::uint64_t{0}));
    }

    // In-place scan: the third pass re-reads the input, so aliasing is only
    // safe because each tile is scanned by the block that reduced it.
    std::vector<std::uint32_t> host(5000, 1);
    DeviceBuffer<std::uint32_t> buffer{std::span<const std::uint32_t>(host)};
    const std::uint64_t total = scan.exclusiveScan(buffer.data(), buffer.data(), host.size(), stream);
    const std::vector<std::uint32_t> actual = buffer.toHost(stream);
    EXPECT_EQ(total, std::uint64_t{5000});
    EXPECT_EQ(actual.back(), 4999u);
}

void testDeviceStateSetAlgebra() {
    Stream stream;
    constexpr StateId kStates = 1000;

    StateSet hostLeft(kStates);
    StateSet hostRight(kStates);
    for (StateId state = 0; state < kStates; state += 2) {
        hostLeft.insert(state);
    }
    for (StateId state = 0; state < kStates; state += 3) {
        hostRight.insert(state);
    }

    DeviceStateSet left(hostLeft, stream);
    const DeviceStateSet right(hostRight, stream);

    EXPECT_EQ(left.count(stream), hostLeft.count());
    EXPECT_EQ(right.count(stream), hostRight.count());
    EXPECT_TRUE(!left.equals(right, stream));

    DeviceStateSet scratch(kStates);
    scratch.assign(left, stream);
    scratch.intersectWith(right, stream);
    StateSet expected = hostLeft;
    expected.intersectWith(hostRight);
    EXPECT_TRUE(scratch.toHost(stream) == expected);
    EXPECT_TRUE(scratch.isSubsetOf(left, stream));

    scratch.assign(left, stream);
    scratch.unionWith(right, stream);
    expected = hostLeft;
    expected.unionWith(hostRight);
    EXPECT_TRUE(scratch.toHost(stream) == expected);
    EXPECT_TRUE(left.isSubsetOf(scratch, stream));

    scratch.assign(left, stream);
    scratch.subtract(right, stream);
    EXPECT_TRUE(!scratch.equals(left, stream));

    // Complement must leave the padding bits of the last word clear.
    scratch.assign(left, stream);
    scratch.complement(stream);
    EXPECT_EQ(scratch.count(stream), kStates - hostLeft.count());
    scratch.complement(stream);
    EXPECT_TRUE(scratch.equals(left, stream));

    DeviceStateSet all(kStates);
    all.fillAll(stream);
    EXPECT_EQ(all.count(stream), kStates);
    all.clear(stream);
    EXPECT_TRUE(all.isEmpty(stream));
}

void testDeviceModelRoundTrip() {
    Stream stream;
    const KripkeStructure structure = testing::randomModel(2048, 3, 5, 31337);
    const DeviceKripke model(structure, stream);

    EXPECT_EQ(model.stateCount(), structure.stateCount());
    EXPECT_EQ(model.transitionCount(), structure.transitionCount());
    EXPECT_TRUE(model.statistics().maxOutDegree > 0);
    EXPECT_TRUE(model.statistics().maxOutDegree <= 3);

    const KripkeStructure downloaded = model.download(stream);
    EXPECT_TRUE(std::ranges::equal(downloaded.rowOffsets(), structure.rowOffsets()));
    EXPECT_TRUE(std::ranges::equal(downloaded.columns(), structure.columns()));
    EXPECT_TRUE(std::ranges::equal(downloaded.labels(), structure.labels()));

    const DeviceKripke copy = model.clone(stream);
    const KripkeStructure cloned = copy.download(stream);
    EXPECT_TRUE(std::ranges::equal(cloned.columns(), structure.columns()));
}

/// The product must satisfy the CSR invariants the model layer relies on:
/// sorted, duplicate-free successor lists and a total relation.
void expectValidCsr(const KripkeStructure& structure, const char* label) {
    bool sorted = true;
    bool total = true;
    for (StateId state = 0; state < structure.stateCount(); ++state) {
        const std::span<const StateId> successors = structure.successors(state);
        total = total && !successors.empty();
        for (std::size_t index = 1; index < successors.size(); ++index) {
            sorted = sorted && successors[index - 1] < successors[index];
        }
    }
    ::kripcuda::testing::expect(sorted, std::string("successor lists sorted and distinct in ") + label,
                                __FILE__, __LINE__);
    ::kripcuda::testing::expect(total, std::string("relation total in ") + label, __FILE__, __LINE__);
}

/// Sequential reference product, used to validate the GPU construction.
KripkeStructure referenceProduct(const KripkeStructure& left, const KripkeStructure& right,
                                 ProductKind kind) {
    const StateId rightStates = right.stateCount();
    const StateId states = left.stateCount() * rightStates;

    KripkeBuilder builder(states, left.propositionCount() + right.propositionCount());
    for (const StateId leftInitial : left.initialStates()) {
        for (const StateId rightInitial : right.initialStates()) {
            builder.addInitialState(leftInitial * rightStates + rightInitial);
        }
    }

    for (StateId leftState = 0; leftState < left.stateCount(); ++leftState) {
        for (StateId rightState = 0; rightState < rightStates; ++rightState) {
            const StateId state = leftState * rightStates + rightState;

            for (PropositionId proposition = 0; proposition < left.propositionCount();
                 ++proposition) {
                builder.setLabel(state, proposition, left.holds(leftState, proposition));
            }
            for (PropositionId proposition = 0; proposition < right.propositionCount();
                 ++proposition) {
                builder.setLabel(state, left.propositionCount() + proposition,
                                 right.holds(rightState, proposition));
            }

            if (kind == ProductKind::Interleaving) {
                for (const StateId target : left.successors(leftState)) {
                    builder.addTransition(state, target * rightStates + rightState);
                }
                for (const StateId target : right.successors(rightState)) {
                    builder.addTransition(state, leftState * rightStates + target);
                }
            } else {
                for (const StateId leftTarget : left.successors(leftState)) {
                    for (const StateId rightTarget : right.successors(rightState)) {
                        builder.addTransition(state, leftTarget * rightStates + rightTarget);
                    }
                }
            }
        }
    }
    return builder.build();
}

void expectProductMatchesReference(const KripkeStructure& left, const KripkeStructure& right,
                                   ProductKind kind, const char* label) {
    Stream stream;
    const DeviceKripke deviceLeft(left, stream);
    const DeviceKripke deviceRight(right, stream);
    const DeviceKripke product = buildProduct(deviceLeft, deviceRight, kind, stream);
    const KripkeStructure built = product.download(stream);
    const KripkeStructure expected = referenceProduct(left, right, kind);

    expectValidCsr(built, label);
    ::kripcuda::testing::expect(std::ranges::equal(built.rowOffsets(), expected.rowOffsets()),
                                std::string("product row offsets match on ") + label, __FILE__,
                                __LINE__);
    ::kripcuda::testing::expect(std::ranges::equal(built.columns(), expected.columns()),
                                std::string("product columns match on ") + label, __FILE__,
                                __LINE__);
    ::kripcuda::testing::expect(std::ranges::equal(built.labels(), expected.labels()),
                                std::string("product labels match on ") + label, __FILE__,
                                __LINE__);
    ::kripcuda::testing::expect(
        std::ranges::equal(product.initialStates(), expected.initialStates()),
        std::string("product initial states match on ") + label, __FILE__, __LINE__);
}

void testProduct() {
    const KripkeStructure mutex = testing::mutualExclusionModel();
    // Label counts either side of a word boundary, so the label merge is
    // exercised both word-aligned and with a bit-level shift.
    const KripkeStructure wide = testing::randomModel(37, 3, 60, 5150);
    const KripkeStructure narrow = testing::randomModel(23, 2, 7, 6161);

    expectProductMatchesReference(mutex, mutex, ProductKind::Interleaving, "mutex^2 interleaving");
    expectProductMatchesReference(mutex, mutex, ProductKind::Synchronous, "mutex^2 synchronous");
    expectProductMatchesReference(wide, narrow, ProductKind::Interleaving, "shifted labels");
    expectProductMatchesReference(narrow, wide, ProductKind::Synchronous, "synchronous labels");
}

/// A model built on the device must be explorable and checkable without ever
/// touching the host: this is the path a real verification run takes.
void testDeviceResidentPipeline() {
    Stream stream;
    const KripkeStructure component = testing::mutualExclusionModel();
    const DeviceKripke deviceComponent(component, stream);
    const DeviceKripke composed =
        buildPower(deviceComponent, 4, ProductKind::Interleaving, stream);

    EXPECT_EQ(composed.stateCount(), 9u * 9u * 9u * 9u);
    EXPECT_EQ(composed.propositionCount(), 8u);

    const ReachabilityResult deviceReachable = computeReachabilityDevice(composed, stream);
    const KripkeStructure downloaded = composed.download(stream);
    const ReachabilityResult hostReachable = computeReachabilityHost(downloaded);
    EXPECT_EQ(deviceReachable.reachableCount, hostReachable.reachableCount);
    EXPECT_TRUE(deviceReachable.levels == hostReachable.levels);

    // Safety of every component of the composition, checked on the product.
    ctl::Formula safety = ctl::constantTrue();
    for (unsigned component_index = 0; component_index < 4; ++component_index) {
        const PropositionId critical0 = component_index * 2;
        const PropositionId critical1 = component_index * 2 + 1;
        safety = safety && ctl::AG(!(ctl::atom(critical0) && ctl::atom(critical1)));
    }

    DeviceCtlEvaluator evaluator(composed, stream);
    const CtlResult device = evaluator.check(safety);
    const CtlResult host = checkCtlHost(downloaded, safety);
    EXPECT_TRUE(device.satisfying == host.satisfying);
    EXPECT_EQ(device.holdsInAllInitialStates, host.holdsInAllInitialStates);
    EXPECT_TRUE(device.holdsInAllInitialStates);
}

} // namespace

int main() {
    if (!hasCudaDevice()) {
        std::cout << "no CUDA device visible; skipping device test suite\n";
        return kripcuda::testing::report("test_device");
    }

    std::cout << "CUDA device: " << cudaDeviceDescription() << '\n';
    testScan();
    testDeviceStateSetAlgebra();
    testDeviceModelRoundTrip();
    testProduct();
    testDeviceResidentPipeline();
    return kripcuda::testing::report("test_device");
}
