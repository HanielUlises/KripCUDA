#include "kripcuda/verification/ctl.hpp"

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kripcuda {
namespace {

/// Sequential CTL evaluator. It mirrors the device evaluator step for step —
/// same fixpoint schedule, same iteration count — so a disagreement between the
/// two can only come from the parallel implementation, never from a difference
/// in semantics.
class HostCtlEvaluator {
public:
    explicit HostCtlEvaluator(const KripkeStructure& structure) : structure_(structure) {}

    const StateSet& evaluate(const ctl::Formula& formula) {
        const auto memoised = memo_.find(formula.identity());
        if (memoised != memo_.end()) {
            return memoised->second;
        }

        StateSet result(structure_.stateCount());
        switch (formula.kind()) {
        case ctl::Formula::Kind::False:
            break;

        case ctl::Formula::Kind::Atom:
            for (StateId state = 0; state < structure_.stateCount(); ++state) {
                if (structure_.holds(state, formula.proposition())) {
                    result.insert(state);
                }
            }
            break;

        case ctl::Formula::Kind::Negation:
            result = evaluate(formula.left());
            result.complement();
            break;

        case ctl::Formula::Kind::Conjunction: {
            const StateSet& left = evaluate(formula.left());
            const StateSet& right = evaluate(formula.right());
            result = left;
            result.intersectWith(right);
            break;
        }

        case ctl::Formula::Kind::Disjunction: {
            const StateSet& left = evaluate(formula.left());
            const StateSet& right = evaluate(formula.right());
            result = left;
            result.unionWith(right);
            break;
        }

        case ctl::Formula::Kind::ExistsNext:
            result = preImage(evaluate(formula.left()));
            break;

        case ctl::Formula::Kind::ExistsUntil: {
            const StateSet& condition = evaluate(formula.left());
            const StateSet& target = evaluate(formula.right());
            for (;;) {
                StateSet next = step(&target, &condition, result);
                ++iterations_;
                if (next == result) {
                    break;
                }
                result = std::move(next);
            }
            break;
        }

        case ctl::Formula::Kind::ExistsGlobally: {
            const StateSet& condition = evaluate(formula.left());
            result = condition;
            for (;;) {
                StateSet next = step(nullptr, &condition, result);
                ++iterations_;
                if (next == result) {
                    break;
                }
                result = std::move(next);
            }
            break;
        }
        }

        return memo_.emplace(formula.identity(), std::move(result)).first->second;
    }

    [[nodiscard]] std::uint64_t iterations() const noexcept { return iterations_; }

private:
    [[nodiscard]] StateSet preImage(const StateSet& target) const {
        StateSet result(structure_.stateCount());
        for (StateId state = 0; state < structure_.stateCount(); ++state) {
            for (const StateId successor : structure_.successors(state)) {
                if (target.contains(successor)) {
                    result.insert(state);
                    break;
                }
            }
        }
        return result;
    }

    /// next = base ∪ (mask ∩ pre∃(current)); a null operand is the neutral
    /// element, as in the device kernel.
    [[nodiscard]] StateSet step(const StateSet* base, const StateSet* mask,
                                const StateSet& current) const {
        StateSet next = preImage(current);
        if (mask != nullptr) {
            next.intersectWith(*mask);
        }
        if (base != nullptr) {
            next.unionWith(*base);
        }
        return next;
    }

    const KripkeStructure& structure_;
    std::unordered_map<const void*, StateSet> memo_;
    std::uint64_t iterations_ = 0;
};

} // namespace

CtlResult checkCtlHost(const KripkeStructure& structure, const ctl::Formula& formula) {
    HostCtlEvaluator evaluator(structure);

    CtlResult result;
    result.satisfying = evaluator.evaluate(formula);
    result.iterations = evaluator.iterations();
    result.holdsInAllInitialStates = true;
    for (const StateId initial : structure.initialStates()) {
        if (!result.satisfying.contains(initial)) {
            result.holdsInAllInitialStates = false;
            break;
        }
    }
    return result;
}

} // namespace kripcuda
