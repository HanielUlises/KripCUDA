#include "kripcuda/verification/formula.hpp"

#include <stdexcept>

namespace kripcuda::ctl {

Formula Formula::make(Kind kind, PropositionId proposition, Formula left, Formula right) {
    auto node = std::make_shared<Node>();
    node->kind = kind;
    node->proposition = proposition;
    node->left = std::move(left.node_);
    node->right = std::move(right.node_);
    return Formula(std::move(node));
}

std::string Formula::toString() const {
    switch (kind()) {
    case Kind::False:
        return "false";
    case Kind::Atom:
        return "p" + std::to_string(proposition());
    case Kind::Negation:
        return "!" + left().toString();
    case Kind::Conjunction:
        return '(' + left().toString() + " && " + right().toString() + ')';
    case Kind::Disjunction:
        return '(' + left().toString() + " || " + right().toString() + ')';
    case Kind::ExistsNext:
        return "EX " + left().toString();
    case Kind::ExistsUntil:
        return "E[" + left().toString() + " U " + right().toString() + ']';
    case Kind::ExistsGlobally:
        return "EG " + left().toString();
    }
    throw std::logic_error("Formula::toString: unhandled operator");
}

Formula constantFalse() {
    return Formula::make(Formula::Kind::False, 0, Formula(nullptr), Formula(nullptr));
}

Formula constantTrue() { return !constantFalse(); }

Formula atom(PropositionId proposition) {
    return Formula::make(Formula::Kind::Atom, proposition, Formula(nullptr), Formula(nullptr));
}

Formula operator!(Formula operand) {
    return Formula::make(Formula::Kind::Negation, 0, std::move(operand), Formula(nullptr));
}

Formula operator&&(Formula left, Formula right) {
    return Formula::make(Formula::Kind::Conjunction, 0, std::move(left), std::move(right));
}

Formula operator||(Formula left, Formula right) {
    return Formula::make(Formula::Kind::Disjunction, 0, std::move(left), std::move(right));
}

Formula implies(Formula antecedent, Formula consequent) {
    return !std::move(antecedent) || std::move(consequent);
}

Formula EX(Formula operand) {
    return Formula::make(Formula::Kind::ExistsNext, 0, std::move(operand), Formula(nullptr));
}

Formula EU(Formula left, Formula right) {
    return Formula::make(Formula::Kind::ExistsUntil, 0, std::move(left), std::move(right));
}

Formula EG(Formula operand) {
    return Formula::make(Formula::Kind::ExistsGlobally, 0, std::move(operand), Formula(nullptr));
}

Formula EF(Formula operand) { return EU(constantTrue(), std::move(operand)); }

Formula AX(Formula operand) { return !EX(!std::move(operand)); }

Formula AF(Formula operand) { return !EG(!std::move(operand)); }

Formula AG(Formula operand) { return !EF(!std::move(operand)); }

Formula AU(Formula left, Formula right) {
    // A[φ U ψ] holds where no run either avoids ψ forever or reaches a state
    // violating both φ and ψ before ψ ever holds.
    const Formula notLeft = !left;
    const Formula notRight = !right;
    return !(EU(notRight, notLeft && notRight) || EG(notRight));
}

} // namespace kripcuda::ctl
