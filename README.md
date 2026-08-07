<p align="center">
  <img src="docs/logo.svg" alt="KripCUDA logo: a directed state graph feeding into a GPU die" width="260">
</p>

<h1 align="center">KripCUDA</h1>

<p align="center">

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.22%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org)
[![Architectures](https://img.shields.io/badge/arch-sm__75%20%7C%20sm__86%20%7C%20sm__89-76B900)](https://developer.nvidia.com/cuda-gpus)
[![Tests](https://img.shields.io/badge/tests-ctest-informational)](tests)

</p>

<p align="center">
GPU-accelerated explicit-state model checking of Kripke structures in C++20 and CUDA.
</p>

<p align="center">
  <img src="docs/example-run.svg" alt="Terminal session: the two-process mutual exclusion example explored on an RTX 3060, followed by the test suite" width="620">
</p>

## Formal setting

A **Kripke structure** over a finite set of atomic propositions *AP* is a tuple

> *M* = (*S*, *S₀*, *R*, *L*)

with *S* a finite set of states, *S₀* ⊆ *S* the initial states, *R* ⊆ *S* × *S* a
*total* transition relation — every state has at least one successor — and
*L*: *S* → 2^*AP* a labelling assigning to each state the propositions true in
it. Totality is what makes every finite prefix extendable to an infinite path,
so the branching-time semantics below are well defined on paths
π = *s*₀*s*₁*s*₂… with (*s*ᵢ, *s*ᵢ₊₁) ∈ *R*.

KripCUDA stores *R* in compressed sparse row form: *R* is the sparse boolean
adjacency matrix of the state graph, and the successor list of a state is one of
its rows. The labelling is a packed bitset — *L*(*s*) as ⌈|*AP*|/64⌉ machine
words — so evaluating a propositional formula at a state is a handful of bitwise
operations, and evaluating it at *all* states is a data-parallel scan over a
dense array.

The library enforces the structural invariants of the definition at construction
time rather than trusting the caller: totality of *R* is checked, duplicate
transitions are removed, and successor lists are sorted, which makes the CSR
layout — and therefore every kernel launch configuration derived from it — a
function of the model alone and not of the order in which the caller happened to
supply its transitions.

## Verification as fixpoint computation

Model checking asks whether *M*, *s* ⊨ φ. For branching-time logics the
computation is a fixpoint over the powerset lattice (2^*S*, ⊆), which is complete
and finite; the modal operators are monotone on it, so by Knaster–Tarski their
least and greatest fixpoints exist and — the lattice having finite height |*S*| —
are reached by Kleene iteration in at most |*S*| steps. Writing
*pre*∃(*X*) = { *s* | ∃*s*′ ∈ *X*. (*s*, *s*′) ∈ *R* } for the existential
pre-image, CTL is characterised by

- ⟦EX φ⟧ = *pre*∃(⟦φ⟧)
- ⟦E φ U ψ⟧ = μ*X*. ⟦ψ⟧ ∪ (⟦φ⟧ ∩ *pre*∃(*X*))
- ⟦EG φ⟧ = ν*X*. ⟦φ⟧ ∩ *pre*∃(*X*)

with the remaining operators derived by duality. Every one of these is a
sequence of *pre*∃ applications interleaved with set operations, and *pre*∃ is
exactly a sparse matrix–vector product over the boolean semiring (∨, ∧) against
*Rᵀ*. The characteristic function of a state set is a bit array over *S*; set
union and intersection are elementwise disjunction and conjunction on it. This
is the reason the whole verification problem is a good fit for a GPU: the inner
operation is an irregular but massively data-parallel SpMV, and the outer loop is
a short sequence of dependent kernel launches whose termination test is a single
scalar reduction.

Reachability, the first of these computations to be implemented here, is the
dual direction: the least fixpoint μ*X*. *S*₀ ∪ *post*∃(*X*) of the image
operator, which is the set of states of *M* that any run can actually occupy.
It is computed frontier-by-frontier — a level-synchronous breadth-first
expansion, one kernel launch per BFS level — which additionally yields the
distance of each state from *S*₀ and hence the diameter of the reachable
fragment. Restricting later verification to that fragment is sound for the
properties that quantify over runs of the system, and is usually a substantial
reduction: in the example above only 8 of the 9 syntactically representable
states are reachable, and it is precisely the unreachable one, (crit, crit),
that would violate mutual exclusion.

The fixpoint characterisation also explains why the parallelisation is safe.
Kleene iteration of a monotone operator converges to the same limit regardless
of the order in which elements are added within an iteration, so the
nondeterministic order in which GPU threads claim states cannot change the
result. In the reachability kernel each state is claimed by a single atomic
compare-and-swap that simultaneously assigns its BFS level; every thread that
could claim a state within a given level writes the same value, so the computed
level array is deterministic and bit-identical to the sequential reference, even
though the frontier order is not.

## The explicit-state trade-off

Explicit-state checking enumerates *S* rather than encoding it symbolically as a
BDD. It surrenders the compression that symbolic methods obtain on structured,
regular models, and pays the full price of the state explosion problem — |*S*|
grows exponentially in the number of concurrent components. What it buys is a
representation whose cost is predictable, whose successor computation is a
memory-bound graph traversal rather than a pointer-chasing operation on a shared
decision-diagram node table, and whose parallelism is not bottlenecked by a
global unique table. On hardware with thousands of concurrently resident threads
and an order of magnitude more memory bandwidth than a host, that trade is
frequently the favourable one, and it becomes more so as the model grows less
regular — exactly the regime in which BDD-based approaches degrade.

The design consequence is that the state graph is materialised once, uploaded
once, and then repeatedly traversed by kernels that never transfer it back:
host–device traffic per verification run is one small scalar per iteration for
the termination test, plus one result vector at the end.

## Architecture

The library keeps model representation, state exploration, property
verification, CUDA execution, and utilities in separate compilation units and
namespaces, with the coupling running in one direction only. Kernels receive a
trivially copyable view of the device-resident structure; ownership of device
memory sits in RAII handles on the host; every CUDA API result is checked and
surfaced as a C++ exception. Public headers are plain C++20 except for those
under `cuda/`, which are meant for translation units compiled by NVCC — the
CUDA-availability queries are deliberately exposed through a header that
includes no CUDA header at all, so host-only consumers never inherit the
toolkit's include path.

Implemented today: the model representation and its builder, reachability
analysis on both host and device, and the CUDA execution layer. The CTL
operators above build directly on the same device model view and label encoding.

## Building

Requires CMake 3.22 or newer, a C++20 host compiler, and CUDA 12 or newer.

```sh
cmake -S . -B build -G Ninja -DKRIPCUDA_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```

`KRIPCUDA_CUDA_ARCHITECTURES` (default `75;86;89`) selects the generated device
code; set it to the compute capability of the target GPU. The device tests and
the example fall back to the host path when no CUDA device is visible, so the
suite is meaningful on machines without one.

The suite validates the device explorer against the sequential reference on a
hand-checkable concurrency model, on pseudo-random models with reachable and
unreachable regions, and on a deep chain that exercises many BFS levels with a
minimal frontier.

## Usage

A model is assembled through `KripkeBuilder`, which validates it and hands back
an immutable `KripkeStructure`:

```cpp
#include "kripcuda/exploration/reachability.hpp"

using namespace kripcuda;

KripkeBuilder builder(stateCount, propositionCount);
builder.addInitialState(0);
builder.addTransition(0, 1);
builder.setLabel(1, criticalSection);
const KripkeStructure model = builder.build();

const ReachabilityResult reachable = computeReachabilityDevice(model);
```

`ReachabilityResult::levels[s]` is the BFS distance of state `s` from the
closest initial state, or `kUnreachable`. The GPU and host explorers produce
identical levels for the reason given above: the fixpoint does not depend on the
order in which states are claimed, even though the frontier order on the GPU is
nondeterministic. `computeReachabilityHost` is the sequential reference and can
be called on the same model when no device is available.

`examples/mutual_exclusion.cpp` builds the two-process model shown at the top of
this page and checks `AG !(crit0 && crit1)` over its reachable fragment.

## Build note

CMake before 3.25 cannot express the CUDA20 dialect for NVCC and derives it from
`CMAKE_CXX_STANDARD`, which fails at generate time. The device sources therefore
live in a separate object library that requests `-std=c++20` from NVCC directly.
