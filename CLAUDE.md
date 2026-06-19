# Scikit-Decide

AI framework for Reinforcement Learning, Automated Planning, and Scheduling. Developed by Airbus AI Research.
Repository: https://github.com/airbus/scikit-decide | Docs: https://airbus.github.io/scikit-decide/

## Quick Reference

```bash
# Build & install (editable, all extras)
uv sync --extra=all -v

# Run Python with the project
uv run python <script.py>

# Run without rebuilding C++ (when only Python changed)
uv run --no-sync python <script.py>

# Tests
uv run pytest tests                          # all tests
uv run pytest tests/path/to/test_file.py -v  # single file
uv run pytest --nbmake notebooks -v          # notebook tests

# Lint (auto-fixes files)
pre-commit run --all-files

# Docs (local dev server)
uv run yarn docs:dev
```

## Architecture

### Core Design: Mixin-Based Capability Composition

Domains and solvers are built by composing orthogonal **builder mixins** across independent dimensions. Each dimension is a single-inheritance chain from most general to most specific.

**Domain dimensions** (each is a mixin chain in `src/skdecide/builders/domain/`):

| Dimension | Chain (general → specific) | Type variable |
|---|---|---|
| Agent | `MultiAgent` → `SingleAgent` | `T_agent` |
| Concurrency | `Parallel` → `Sequential` | `T_concurrency` |
| Dynamics | `Environment` → `Simulation` → `UncertainTransitions` → `EnumerableTransitions` → `DeterministicTransitions` | — |
| Events | `Events` → `Actions` → `UnrestrictedActions` | — |
| Memory | `History` → `FiniteHistory` → `Markovian` → `Memoryless` | `T_memory` |
| Observability | `PartiallyObservable` → `TransformedObservable` → `FullyObservable` | — |
| Value | `Rewards` → `PositiveCosts` | — |
| Initialization | `Initializable` → `UncertainInitialized` → `DeterministicInitialized` | — |
| Goals | `Goals` (opt-in) | — |
| Constraints | `Constrained` (opt-in) | — |
| Renderability | `Renderable` (opt-in) | — |

**Domain presets** (common combinations in `src/skdecide/domains.py`):
- `Domain` — most general (MultiAgent, Parallel, Environment, History, PartiallyObservable, Rewards)
- `RLDomain` — RL (SingleAgent, Sequential, Environment, Actions, Markovian, TransformedObservable, Rewards)
- `MDPDomain` — MDP (adds EnumerableTransitions, FullyObservable, DeterministicInitialized)
- `GoalMDPDomain` — Goal MDP (adds Goals, PositiveCosts)
- `DeterministicPlanningDomain` — classical planning (DeterministicTransitions, Goals, PositiveCosts)
- `POMDPDomain` / `GoalPOMDPDomain` — partially observable variants
- `StatelessSimulatorDomain` — external simulation wrapper

**Solver capability mixins** (in `src/skdecide/builders/solver/`):
- Solvability: `FromInitialState`, `FromAnyState`
- Policy: `Policies` → `UncertainPolicies` → `DeterministicPolicies`
- Assessment: `Utilities`, `QValues`
- Restoration: `Restorable`
- Parallelization: `ParallelSolver`

### Method Naming Convention (Critical Pattern)

Three-tier method visibility used throughout the codebase:

```
domain.get_X()        # Public API — autocast wrapper, user calls this
domain._get_X()       # LRU-cached middle layer — calls _get_X_()
domain._get_X_()      # Implementation point — override this in subclasses
                      # Trailing _ means "result is constant, safe to cache"
```

For domain dynamics:
```
domain.step(action)          # Public — manages memory, observation, autocast
domain._state_step(action)   # Override point — implement transition logic here
domain.reset()               # Public — calls _state_reset()
domain._state_reset()        # Override point — implement initialization here
```

### Autocast System

The `@autocastable` decorator + `autocast_all()` function enable automatic type conversion between domains and solvers. When a solver declares `T_domain`, the framework:
1. Casts the domain up to the solver's abstraction level (domain → solver)
2. Casts solver outputs down to the domain's concrete types (solver → domain)

Type parameters reference `D.T_state`, `D.T_observation`, `D.T_event`, `D.T_value`, `D.T_predicate`, `D.T_info`.

### Solver-Domain Compatibility

Every solver declares a `T_domain` class attribute specifying required domain capabilities. The framework introspects this via MRO to extract builder requirements. `Solver.check_domain(domain)` validates that a domain satisfies all requirements.

Solvers receive a `domain_factory: Callable[[], Domain]` (not a domain instance) to support creating multiple copies for parallel solving.

### Plugin System (Entry Points)

Domains and solvers register via Python entry points in `pyproject.toml`:
```toml
[project.entry-points."skdecide.domains"]
MyDomain = "skdecide.hub.domain.my_domain:MyDomain [domains]"

[project.entry-points."skdecide.solvers"]
MySolver = "skdecide.hub.solver.my_solver:MySolver [solvers]"
```

Discovered at runtime via `skdecide.utils.get_registered_domains()` / `get_registered_solvers()` and `load_registered_domain(name)` / `load_registered_solver(name)`.

## Source Layout

```
src/skdecide/
├── core.py                    # Type system (D, Space, Distribution, autocast)
├── domains.py                 # Domain presets (RLDomain, MDPDomain, etc.)
├── solvers.py                 # Solver base class + domain factory wrapping
├── utils.py                   # Plugin registry, match_solvers(), utilities
├── builders/
│   ├── domain/                # Domain capability mixins (13 files)
│   │   ├── dynamics.py        # Environment → ... → DeterministicTransitions
│   │   ├── events.py          # Events → Actions → UnrestrictedActions
│   │   ├── observability.py   # PartiallyObservable → FullyObservable
│   │   ├── memory.py          # History → Markovian → Memoryless
│   │   ├── goals.py, value.py, agent.py, concurrency.py, ...
│   │   └── scheduling/        # Scheduling-specific builders (20 files)
│   └── solver/                # Solver capability mixins
│       ├── policy.py          # Policies → DeterministicPolicies
│       ├── assessability.py   # Utilities, QValues
│       └── fromanystatesolvability.py, restorability.py, parallelability.py
└── hub/
    ├── domain/                # Concrete domain implementations
    │   ├── gym/               # Gymnasium wrappers
    │   ├── maze/              # Maze toy domain
    │   ├── rcpsp/             # Scheduling (RCPSP variants)
    │   ├── flight_planning/   # Flight planning (largest, 58 files)
    │   ├── up/                # Unified Planning bridge
    │   ├── rddl/              # RDDL domains
    │   └── ...
    ├── solver/                # Concrete solver implementations
    │   ├── astar/             # A* search
    │   ├── mcts/              # Monte Carlo Tree Search
    │   ├── stable_baselines/  # Stable-Baselines3 wrapper
    │   ├── ray_rllib/         # Ray RLlib wrapper
    │   ├── do_solver/         # Discrete Optimization wrapper
    │   ├── cgp/               # Cartesian Genetic Programming
    │   ├── up/                # Unified Planning solvers
    │   └── ... (26 solvers total)
    └── space/gym/             # Gymnasium space definitions

cpp/                           # C++ source (performance-critical solvers)
├── src/hub/solver/            # C++ solvers: Astar, AOstar, BFWS, IW, RIW, LRTDP, MCTS, MARTDP, ILAOstar
├── src/builders/domain/       # C++ domain builders
├── sdk/                       # Third-party: pybind11, spdlog, json, Catch2
└── CMakeLists.txt             # CMake build (C++20, OpenMP/TBB)

tests/                         # pytest tests (71 files)
├── autocast/                  # Autocast system tests
├── domains/                   # Domain tests
├── solvers/{python,cpp}/      # Solver tests
├── scheduling/                # Scheduling tests
└── flight_planning/           # Flight planning tests

notebooks/                     # Tutorial Jupyter notebooks (15)
examples/                      # Example scripts (153 files)
docs/                          # VuePress documentation source
```

## How to Add a New Domain

1. Create `src/skdecide/hub/domain/my_domain/__init__.py` re-exporting the class
2. Define the domain class composing appropriate builder mixins:
```python
from skdecide import DeterministicPlanningDomain, UnrestrictedActions, Space, Value
from skdecide.builders.domain import Renderable

class MyDomain(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = MyState        # Define state type
    T_observation = T_state  # FullyObservable → obs = state
    T_event = MyAction       # Action type

    def _state_reset(self) -> MyState: ...              # Initial state
    def _get_next_state(self, memory, action) -> MyState: ...  # Transition
    def _get_transition_value(self, memory, action, next_state) -> Value[float]: ...
    def _is_terminal(self, state) -> bool: ...           # Goal check
    def _get_action_space_(self) -> Space[MyAction]: ... # Available actions
    def _get_observation_space_(self) -> Space[MyState]: ...
    def _get_goals_(self) -> Space[MyState]: ...         # Goal states
    def _get_applicable_actions_from(self, memory) -> Space[MyAction]: ...
```
3. Register in `pyproject.toml` entry points:
```toml
[project.entry-points."skdecide.domains"]
MyDomain = "skdecide.hub.domain.my_domain:MyDomain [domains]"
```

## How to Add a New Python-Only Solver

1. Create `src/skdecide/hub/solver/my_solver/__init__.py` re-exporting the class
2. Define the solver with `T_domain` specifying required domain capabilities:
```python
from skdecide import Solver, Domain, hub
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, ...
from skdecide.builders.solver import DeterministicPolicies

class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals,
        Markovian, FullyObservable, PositiveCosts):
    pass

class MySolver(Solver, DeterministicPolicies):
    T_domain = D

    def __init__(self, domain_factory, ...):
        super().__init__(domain_factory=domain_factory)
        ...

    def _solve(self) -> None: ...                               # Run solver
    def _get_next_action(self, observation) -> D.T_event: ...   # Policy query
    def _is_solution_defined_for(self, observation) -> bool: ...
```
3. Register in `pyproject.toml` entry points.

## How to Add a New C++ Solver

All C++ solvers follow the same architecture. Use A* as the reference template for simple solvers, MCTS for complex multi-parameter solvers.

### Naming Convention (Solvers and Directories)

All solver names use the algorithm's **acronym** consistently across every layer:

| Layer | Pattern | Example (Value Iteration) |
|---|---|---|
| Directory (C++) | `cpp/src/hub/solver/{acronym}/` | `cpp/src/hub/solver/vi/` |
| Directory (Python) | `src/skdecide/hub/solver/{acronym}/` | `src/skdecide/hub/solver/vi/` |
| C++ header file | `{acronym}.hh` | `vi.hh` |
| C++ impl file | `impl/{acronym}_impl.hh` | `impl/vi_impl.hh` |
| C++ pybind header | `py_{acronym}.hh` | `py_vi.hh` |
| C++ pybind source | `py_{acronym}.cc` | `py_vi.cc` |
| C++ template inst. | `impl/py_{acronym}_solver.cc.in` | `impl/py_vi_solver.cc.in` |
| C++ class name | `{Acronym}Solver` | `VISolver` |
| C++ pybind class | `Py{Acronym}Solver` | `PyVISolver` |
| C++ domain alias | `Py{Acronym}Domain` | `PyVIDomain` |
| Pybind module name | `_{Acronym}Solver_` | `_VISolver_` |
| Init function | `init_py{acronym}` | `init_pyvi` |
| CMake library | `py_{acronym}` | `py_vi` |
| Python class name | `{ACRONYM}` | `VI` |
| Python file | `{acronym}.py` | `vi.py` |
| pyproject.toml key | `{ACRONYM}` | `VI` |
| Include guard | `SKDECIDE_{ACRONYM}_HH` | `SKDECIDE_VI_HH` |

Existing solver acronyms: `aostar`, `astar`, `bfws`, `despot`, `idual`, `ilaostar`, `iw`, `lrtdp`, `martdp`, `mcts`, `riw`, `sarsop`, `witness`.

### C++ Solver Architecture Overview

```
Python Layer                    C++ Layer
────────────────────────────────────────────────────────────
hub/solver/{acronym}/           cpp/src/hub/solver/{acronym}/
  __init__.py                     {acronym}.hh         (algorithm template)
  {acronym}.py  ──imports──→      impl/{acronym}_impl.hh (algorithm implementation)
    _{Acronym}Solver_             py_{acronym}.hh       (pybind11 wrapper)
                                  py_{acronym}.cc       (module registration)
                                  impl/py_{acronym}_solver.cc.in (template instantiation)
                                  CMakeLists.txt
```

### C++ Algorithm Code Independence from Python

**Critical rule**: Algorithm headers (`{acronym}.hh`) and implementation files (`impl/{acronym}_impl.hh`) must **never** include `<pybind11/pybind11.h>` or depend on Python-specific types. They are pure C++ templates that work with any domain type (not just `PythonDomainProxy`). Only the pybind wrapper files (`py_{acronym}.hh`, `py_{acronym}.cc`) may include pybind11.

**Comparing domain objects (State, Action, etc.)**: The proxy types (`State`, `Action`, `Value`, etc.) all inherit from `PyObj`, which provides `Hash` and `Equal` inner structs. Use these instead of raw `operator==` or `pyobj().equal()`:

```cpp
// WRONG — depends on pybind11, breaks non-Python domains:
if (an->action.pyobj().equal(target.pyobj())) { ... }

// CORRECT — uses the type's own Equal, works with any domain:
if (typename Action::Equal()(an->action, target)) { ... }
```

`PyObj::Hash` delegates to `PythonHash<Texecution>` and `PyObj::Equal` delegates to `PythonEqual<Texecution>` (defined in `python_hash_eq.hh`). These handle GIL acquisition, Python `__eq__`/`__hash__`, numpy arrays, and fallback to `__repr__`/`__str__` — all without exposing pybind11 to the algorithm code.

The same `Hash` and `Equal` types are used by `SetTypeDeducer` and `MapTypeDeducer` (in `associative_container_deducer.hh`) to build the solver's internal hash sets and maps.

### File-by-File Guide

#### 1. Algorithm Header: `mysolver.hh`

Declare the solver as a class template. Use `#define` macros to avoid repeating long template parameter lists in the implementation file.

```cpp
namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class MySolver {
public:
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Value Value;
    typedef Texecution_policy ExecutionPolicy;

    // Functors for Python-provided callbacks
    typedef std::function<bool(Domain &, const State &)> GoalCheckerFunctor;
    typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
    typedef std::function<bool(const MySolver &, Domain &)> CallbackFunctor;

    MySolver(Domain &domain,
             const GoalCheckerFunctor &goal_checker,
             const HeuristicFunctor &heuristic,
             const CallbackFunctor &callback = [](const MySolver &, Domain &) { return false; },
             bool verbose = false);

    void clear();
    void solve(const State &s);
    bool is_solution_defined_for(const State &s) const;
    const Action &get_best_action(const State &s) const;
    Value get_best_value(const State &s) const;

private:
    Domain &_domain;
    GoalCheckerFunctor _goal_checker;
    HeuristicFunctor _heuristic;
    CallbackFunctor _callback;
    // ... solver-specific data structures
};

} // namespace skdecide
```

**Convention**: Use `ExecutionPolicy::template atomic<T>` for thread-safe fields and `ExecutionPolicy::Mutex` for locks when supporting `ParallelExecution`.

#### 2. Algorithm Implementation: `impl/mysolver_impl.hh`

Define template macros to avoid repeating the full parameter list, then implement methods:

```cpp
#define SK_MYSOLVER_TEMPLATE_DECL \
  template <typename Tdomain, typename Texecution_policy>

#define SK_MYSOLVER_CLASS MySolver<Tdomain, Texecution_policy>

SK_MYSOLVER_TEMPLATE_DECL
SK_MYSOLVER_CLASS::MySolver(Domain &domain,
                            const GoalCheckerFunctor &goal_checker,
                            const HeuristicFunctor &heuristic,
                            const CallbackFunctor &callback,
                            bool verbose)
    : _domain(domain), _goal_checker(goal_checker),
      _heuristic(heuristic), _callback(callback) {}

SK_MYSOLVER_TEMPLATE_DECL
void SK_MYSOLVER_CLASS::solve(const State &s) {
    // Algorithm implementation using _domain methods:
    //   _domain.get_applicable_actions(s)
    //   _domain.get_next_state(s, action)
    //   _domain.get_transition_value(s, action, next_state)
    //   _domain.is_terminal(state)
    //   _domain.is_goal(state)
}
```

#### 3. Pybind11 Wrapper: `py_mysolver.hh`

This is the bridge between Python and C++. It follows a strict pattern with three layers:

```cpp
namespace skdecide {

// Domain alias for this solver
template <typename Texecution>
using PyMySolverDomain = PythonDomainProxy<Texecution>;

class PyMySolver {
public:
    PyMySolver(py::object &solver, py::object &domain,
               const std::function<py::object(const py::object &, const py::object &)> &goal_checker,
               const std::function<py::object(const py::object &, const py::object &)> &heuristic,
               bool parallel = false,
               const std::function<py::bool_(const py::object &)> &callback = nullptr,
               bool verbose = false);

    void close();
    void clear();
    void solve(const py::object &s);
    py::bool_ is_solution_defined_for(const py::object &s);
    py::object get_next_action(const py::object &s);
    py::object get_utility(const py::object &s);

private:
    // === LAYER 1: Abstract base interface ===
    class BaseImplementation {
    public:
        virtual ~BaseImplementation() {}
        virtual void close() = 0;
        virtual void clear() = 0;
        virtual void solve(const py::object &s) = 0;
        virtual py::bool_ is_solution_defined_for(const py::object &s) = 0;
        virtual py::object get_next_action(const py::object &s) = 0;
        virtual py::object get_utility(const py::object &s) = 0;
    };

    // === LAYER 2: Template implementation (one per Texecution) ===
    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public:
        Implementation(py::object &solver, py::object &domain, /* ... args ... */)
        {
            // Store Python solver reference
            _pysolver = std::make_unique<py::object>(solver);

            // Wrap Python domain in C++ proxy
            check_domain(domain);
            _domain = std::make_unique<PyMySolverDomain<Texecution>>(domain);

            // Create C++ solver, wrapping Python callbacks as C++ functors:
            _solver = std::make_unique<MySolver<PyMySolverDomain<Texecution>, Texecution>>(
                *_domain,
                // Goal checker: Python → C++ conversion
                [this](PyMySolverDomain<Texecution> &d,
                       const typename PyMySolverDomain<Texecution>::State &s) {
                    auto fgc = [this](const py::object &dd, const py::object &ss,
                                      [[maybe_unused]] const py::object &ii) {
                        return _goal_checker(dd, ss);
                    };
                    std::unique_ptr<py::object> r = d.call(nullptr, fgc, s.pyobj());
                    typename GilControl<Texecution>::Acquire acquire;
                    return r->template cast<bool>();
                },
                // Heuristic: Python → C++ conversion
                [this](PyMySolverDomain<Texecution> &d,
                       const typename PyMySolverDomain<Texecution>::State &s) {
                    auto fh = [this](const py::object &dd, const py::object &ss,
                                     [[maybe_unused]] const py::object &ii) {
                        return _heuristic(dd, ss);
                    };
                    return typename PyMySolverDomain<Texecution>::Value(
                        d.call(nullptr, fh, s.pyobj()));
                },
                verbose);
        }

        // Method implementations delegate to _solver, with GIL management:
        virtual void solve(const py::object &s) {
            typename GilControl<Texecution>::Release release;  // Release GIL during C++ work
            _solver->solve(typename PyMySolverDomain<Texecution>::State(s));
        }

        virtual py::object get_next_action(const py::object &s) {
            return _solver->get_best_action(
                typename PyMySolverDomain<Texecution>::State(s)).pyobj();
        }

    private:
        std::unique_ptr<py::object> _pysolver;
        std::unique_ptr<PyMySolverDomain<Texecution>> _domain;
        std::unique_ptr<MySolver<PyMySolverDomain<Texecution>, Texecution>> _solver;
        // ... stored Python callbacks
    };

    // === LAYER 3: Runtime execution selector ===
    struct ExecutionSelector {
        bool _parallel;
        template <typename Propagator> struct Select {
            template <typename... Args>
            Select(ExecutionSelector &s, Args... args) {
                if (s._parallel)
                    Propagator::template PushType<ParallelExecution>::Forward(args...);
                else
                    Propagator::template PushType<SequentialExecution>::Forward(args...);
            }
        };
    };

    struct SolverInstantiator {
        std::unique_ptr<BaseImplementation> &_implementation;
        template <typename... TypeInstantiations> struct Instantiate {
            Instantiate(SolverInstantiator &This, /* constructor args */) {
                This._implementation = std::make_unique<
                    Implementation<TypeInstantiations...>>(/* args */);
            }
        };
    };

    std::unique_ptr<BaseImplementation> _implementation;
};

} // namespace skdecide
```

**Key patterns**:
- `GilControl<Texecution>::Release` before entering C++ computation
- `GilControl<Texecution>::Acquire` before calling Python or casting py::object
- `d.call(thread_id, lambda, args...)` to invoke Python domain methods through the proxy
- `s.pyobj()` to extract the underlying `py::object` from a C++ State wrapper

#### 4. Module Registration: `py_mysolver.cc`

```cpp
#include "py_mysolver.hh"

namespace skdecide {

void init_pymysolver(py::module &m) {
    py::class_<PyMySolver> py_solver(m, "_MySolver_");
    py_solver
        .def(py::init<py::object &, py::object &,
             const std::function<py::object(const py::object &, const py::object &)> &,
             const std::function<py::object(const py::object &, const py::object &)> &,
             bool,
             const std::function<py::bool_(const py::object &)> &,
             bool>(),
             py::arg("solver"), py::arg("domain"),
             py::arg("goal_checker"), py::arg("heuristic"),
             py::arg("parallel") = false,
             py::arg("callback") = nullptr,
             py::arg("verbose") = false)
        .def("close", &PyMySolver::close)
        .def("clear", &PyMySolver::clear)
        .def("solve", &PyMySolver::solve, py::arg("state"))
        .def("is_solution_defined_for", &PyMySolver::is_solution_defined_for)
        .def("get_next_action", &PyMySolver::get_next_action)
        .def("get_utility", &PyMySolver::get_utility);
}

} // namespace skdecide
```

#### 5. Template Instantiation: `impl/py_mysolver_solver.cc.in`

This `.cc.in` file is processed by CMake to generate explicit template instantiations for each execution policy combination:

```cpp
#include "${CMAKE_SOURCE_DIR}/src/hub/solver/mysolver/mysolver.hh"
#include "${CMAKE_SOURCE_DIR}/src/hub/solver/mysolver/impl/mysolver_impl.hh"
#include "${CMAKE_SOURCE_DIR}/src/utils/python_domain_proxy.hh"

template class skdecide::MySolver<skdecide::PythonDomainProxy<${Texecution}>,
                                  ${Texecution}>;
```

CMake substitutes `${Texecution}` with each variant and generates separate `.cc` files (e.g., `py_mysolver_solver_Seq.cc`, `py_mysolver_solver_Par.cc`).

#### 6. CMakeLists.txt

```cmake
IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_MYSOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_mysolver_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par")

    ADD_LIBRARY(py_mysolver STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/py_mysolver.cc
        ${PYTHON_MYSOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_mysolver PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_mysolver ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_mysolver)
ENDIF ()
```

#### 7. Registration Steps

After creating the C++ files:

1. Add `ADD_SUBDIRECTORY(mysolver)` to `cpp/src/hub/solver/CMakeLists.txt`
2. Add `init_pymysolver(m);` to `cpp/src/hub/py_skdecide.cc` (and declare `void init_pymysolver(py::module &m);` at the top)
3. Create the Python wrapper `src/skdecide/hub/solver/mysolver/mysolver.py` that imports `from skdecide.hub.__skdecide_hub_cpp import _MySolver_`
4. Register in `pyproject.toml` entry points

### Template Instantiation System (`GenerateTemplateInstantiationFiles.cmake`)

The CMake function `generate_template_instantiation_files()` generates explicit template instantiation `.cc` files from `.cc.in` templates:

```cmake
generate_template_instantiation_files(
    OUTPUT_VARIABLE             # CMake variable to store generated file paths
    "path/to/template.cc.in"   # Template file with ${VarName} placeholders
    "${CMAKE_CURRENT_BINARY_DIR}"  # Output directory
    "VarName1" "ClassA!ShortA;ClassB!ShortB"  # Param 1: name + semicolon-separated values
    "VarName2" "ClassX!ShortX;ClassY"         # Param 2: name + values
    ...
)
```

It generates the **Cartesian product** of all parameter values. Each value can have an optional `!ShortName` suffix used in the generated filename. Generated files are named `{template}_{Short1}{Short2}...{ShortN}.cc`.

**Common template parameters across all solvers**:

| Parameter | Values | Short names |
|---|---|---|
| `Texecution` | `SequentialExecution`, `ParallelExecution` | `Seq`, `Par` |
| `Thashing_policy` | `StateFeatureHash`, `DomainStateHash` | `Sfa`, `Dsh` |
| `Trollout_policy` | `SimulationRollout`, `EnvironmentRollout` | `Sim`, `Env` |

**Instantiation counts by solver**:

| Solver | Template params | Generated files |
|---|---|---|
| Astar, AOstar, ILAOstar, LRTDP | Texecution (2) | 2 |
| BFWS, IW | Texecution × Thashing (2×2) | 4 |
| RIW | Texecution × Thashing × Trollout (2×2×2) | 8 |
| MARTDP | Texecution (1, Seq only) | 1 |
| MCTS | 8 params (2×3×1×2×2×2×2×1) | 192 + 12 partial |

### C++ Domain Proxy System

Solvers interact with Python domains through `PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>` which provides:

- **State/Action/Value types** — wrappers around `py::object` with GIL-safe access
- **Domain methods** — `get_applicable_actions()`, `get_next_state()`, `get_transition_value()`, `is_terminal()`, `is_goal()`, `reset()`, `step()`, `sample()`
- **Parallel execution support** — NNG sockets for worker pool pattern
- **Agent data access** — `SingleAgent` (direct) or `MultiAgent` (dict-keyed)

Most solvers use the simplified alias:
```cpp
template <typename Texecution>
using PyMySolverDomain = PythonDomainProxy<Texecution>;
// Defaults: SingleAgent, FullyObservable, FullyControllable, Markovian
```

### Execution Model

`SequentialExecution` and `ParallelExecution` (in `cpp/src/utils/execution.hh`) provide:

| Feature | SequentialExecution | ParallelExecution |
|---|---|---|
| `atomic<T>` | Plain `T` | `std::atomic<T>` |
| `Mutex` | No-op | `std::mutex` or OMP lock |
| `protect(f)` | Direct call | `std::scoped_lock` |
| GIL handling | Direct Python calls | Release/Acquire around C++/Python boundary |

### Python Process ↔ C++ Thread Communication (Critical Architecture)

When `parallel=True`, C++ solver threads need to call Python domain methods across process boundaries. This is the core IPC protocol:

**Architecture**: C++ threads cannot hold the Python GIL simultaneously, so each parallel domain worker runs in a **separate Python process** (via `multiprocessing`). Communication uses a dual-channel protocol:

```
C++ Solver Thread                    Python Worker Process
─────────────────────                ──────────────────────
1. [Acquire GIL]
2. Call domain.launch(func, args, thread_id)
   → returns worker domain_id
3. [Release GIL]
                                     4. Worker receives job (via Pipe or SharedMemory)
                                     5. Executes: domain.func(*args)
                                     6. Stores result in _job_results[domain_id]
                                     7. Sends NNG Push notification (byte "0" or error)
                                        ↓
8. [Block on NNG Pull socket]  ←─────┘
   recv_msg() → check success/error
9. [Acquire GIL]
10. result = domain.get_result(domain_id)
11. [Release GIL]
12. Return result to solver
```

**IPC Channels**:
- **Job dispatch**: Python `multiprocessing.Pipe` (PipeParallelDomain) or shared memory + condition variables (ShmParallelDomain)
- **Completion notification**: NNG push/pull sockets over `ipc:///tmp/tmpXXXXXX.ipc`
  - Python side: `pynng.Push0()` sends notification after job completion
  - C++ side: `nng::pull::open()` blocks until notification received
- **Result retrieval**: Python `Manager` list or shared memory arrays

**Python side** (`src/skdecide/parallel_domains.py`):
- `PipeParallelDomain` — uses `mp.Pipe()` for request/response
- `ShmParallelDomain` — uses shared memory arrays with condition variables
- Worker processes run `_launch_domain_server_()` which loops: receive job → execute → notify → repeat
- Domain factory creates one domain instance per worker process

**C++ side** (`cpp/src/utils/impl/python_domain_proxy_call_impl.hh`):
- `SequentialExecution`: Direct Python calls, no IPC needed
- `ParallelExecution`: Opens NNG pull sockets in constructor, blocks on `conn->recv_msg()` during `do_launch()`

**GIL management** (`cpp/src/utils/python_gil_control.hh`):
- `GilControl<ParallelExecution>::Acquire` — locks a recursive mutex then acquires GIL (serializes GIL access across C++ threads)
- `GilControl<ParallelExecution>::Release` — releases GIL so Python workers can execute
- `GilControl<SequentialExecution>` — no-ops (GIL already held)

**Dependencies**: `pynng` (Python, for Push0 sockets), `nng` + `nngpp` (C++, for pull sockets). The `nngpp` C++ wrapper is unmaintained (last update 2020) — a future migration to a modern IPC library is planned.

**Key files**:
- `src/skdecide/parallel_domains.py` — Python worker processes and IPC setup
- `cpp/src/utils/impl/python_domain_proxy_call_impl.hh` — C++ parallel call implementation
- `cpp/src/utils/impl/python_domain_proxy_impl.hh` — NNG socket initialization
- `cpp/src/utils/python_gil_control.hh` — GIL acquisition/release wrappers
- `cpp/src/utils/python_globals.hh` — cached Python globals (avoids GIL in hot paths)

### Python Wrapper Pattern

The Python solver class (`mysolver.py`) wraps the C++ `_MySolver_` class:

```python
from skdecide.hub.__skdecide_hub_cpp import _MySolver_ as mysolver_cpp

class MySolver(Solver, DeterministicPolicies):
    T_domain = D  # declare domain requirements

    def __init__(self, domain_factory, goal_checker=None, heuristic=None,
                 parallel=False, verbose=False, callback=None):
        super().__init__(domain_factory=domain_factory)
        # ... store params

    def _solve(self) -> None:
        self._solver = mysolver_cpp(
            self, self._domain,
            goal_checker=lambda d, s: self._goal_checker(d, s),
            heuristic=lambda d, s: self._heuristic(d, s),
            parallel=self._parallel,
            callback=lambda slv: self._callback(self),  # pass Python solver, not C++ one
            verbose=self._verbose)
        self._solver.solve(self._domain.reset())

    def _get_next_action(self, observation):
        action = self._solver.get_next_action(observation)
        return action

    def _is_solution_defined_for(self, observation):
        return self._solver.is_solution_defined_for(observation)
```

## How to Use Scikit-Decide (Client Code)

```python
from skdecide import utils

# Discover available solvers/domains
print(utils.get_registered_solvers())
print(utils.get_registered_domains())

# Load and instantiate
MyDomain = utils.load_registered_domain("Maze")
MySolver = utils.load_registered_solver("Astar")

# Check compatibility
assert MySolver.check_domain(MyDomain())

# Solve
domain = MyDomain()
with MySolver(domain_factory=lambda: MyDomain()) as solver:
    solver.solve()
    # Query policy
    obs = domain.reset()
    action = solver.sample_action(obs)
    outcome = domain.step(action)
```

## Scheduling Subsystem

Located in `src/skdecide/builders/domain/scheduling/` (20 files). Provides rich modeling for Resource-Constrained Project Scheduling (RCPSP) and variants:
- Task modeling: duration, progress, modes, skills, conditional tasks
- Resource modeling: types, availability, consumption, renewability, costs
- Constraints: precedence, time lags, time windows, preemptivity, preallocations
- Hub domains: `RCPSP`, `MRCPSP`, `RCPSPCalendar`, `Stochastic_RCPSP`, etc.
- Hub solvers: `DOSolver` (discrete optimization), `PilePolicy`, meta-policy scheduling

## Build System

- **Python**: `pyproject.toml` with `scikit-build-core` backend
- **C++**: CMake (C++20), pybind11 bindings, OpenMP or TBB for parallelism
- **Compiled extension**: `__skdecide_hub_cpp` installed into `skdecide/hub/`
- **Python 3.10+** required, 3.12 recommended
- **macOS prerequisites**: `brew install cmake boost libomp` + environment variables for OpenMP

## Dependencies

- **Core**: `pynng`, `pathos`, `discrete-optimization`
- **Shared**: `gymnasium`, `numpy`, `scipy`, `unified-planning`
- **Domains extra**: `matplotlib`, `openap`, `cartopy`, `pygrib`, `pyRDDLGym`, `plado`
- **Solvers extra**: `ray[rllib]`, `stable-baselines3`, `torch`, `jax`, `openevolve`, UP engines
- **Dev/test**: `pytest`, `pytest-cases`, `nbmake`, `optuna`, `jupyter`

## Code Style

- Python: ruff (import sorting + unused import removal + formatting)
- C++: clang-format (`.clang-format` config)
- Notebooks: stripped outputs via nbstripout
- Pre-commit hooks enforce all of the above
- MIT license header in all source files

### C++ Coding Standards

- **No magic numbers**: All algorithm parameters (epsilon, discount, max_iterations, lp_infinity, terminal_value, dead_end_cost, etc.) must be constructor parameters with sensible defaults. Never hardcode tuning constants in the algorithm body.
- **Value objects for costs/rewards**: All rewards, costs, heuristic values, and transition values must be passed and stored as `Value` objects (the domain's `Value` type), not raw `double` or `float`. The `Value` type carries both the numeric cost/reward and metadata (e.g., boolean flags). Functors returning costs or heuristics must return `Value`, not numeric primitives.
- **Default member initializers**: Members used by algorithm internals (not constructor parameters) should use C++11 default member initializers (`bool _initialized = false;`) to prevent undefined behavior from uninitialized reads.
- **Constrained domain support**: Use the `has_get_constraints<Domain>` SFINAE trait with `if constexpr` to optionally support constrained domains. This pattern avoids separate solver classes for constrained vs unconstrained variants:
  ```cpp
  template <typename T, typename = void>
  struct has_get_constraints : std::false_type {};
  template <typename T>
  struct has_get_constraints<T, std::void_t<decltype(std::declval<T>().get_constraints())>>
      : std::true_type {};

  // In solver methods:
  if constexpr (has_get_constraints<Domain>::value) {
      auto constraints = _domain.get_constraints();
      // handle secondary costs, C11 constraints, etc.
  }
  ```

## Testing Conventions

- pytest with `pytest-cases` for parametrized fixtures
- Tests organized by subsystem: `autocast/`, `domains/`, `solvers/{python,cpp}/`, `scheduling/`, `flight_planning/`
- CI splits tests to avoid OpenMP conflicts between C++ solvers and deep-learning frameworks
- Notebook testing via `nbmake`

## Advanced C++ Solver Patterns

### Solver Specialization (e.g., IDAstar from LDFS, LRTAstar from LRTDP)

When an algorithm is a specialization of another (e.g., IDAstar is LDFS on deterministic domains), define it in the same directory:

- C++ side: `IDAstarSolver` inherits from `LDFSSolver` in the same `ldfs.hh`/`ldfs_impl.hh`
- Template instantiation: add `IDAstarSolver` to the same `.cc.in` file
- Pybind: template the `Implementation` on a solver type tag; use `std::conditional_t` to select the right solver type
- Python side: separate class in the same `.py` file, separate entry point in `pyproject.toml`
- The specialized solver may omit irrelevant parameters and add its own methods (e.g., `get_plan()`)

### Meta-Solvers with Template Template Parameters (e.g., SSiPP)

When a solver wraps an inner solver (meta-solver pattern), use template template parameters:

```cpp
template <typename Tdomain, typename Texecution_policy,
          template <typename, typename> class TinnerSolver>
class SSiPPSolver { ... };
```

**Key patterns:**
- Use `if constexpr` + `std::is_same_v` in the constructor to handle inner solver constructor differences (e.g., LRTDP needs thread_id in functors, ILAOstar doesn't)
- Store a type-erased factory (`std::function`) that captures inner solver extra args via `std::apply` on a `std::tuple`
- The variadic constructor template captures inner args: `template <typename... InnerSolverArgs> SSiPPSolver(..., InnerSolverArgs&&... args)`
- If the constructor is a member template, include the impl header in `py_{solver}.hh` so the definition is visible for instantiation
- Pybind layer: use tag types (e.g., `LRTDPInnerSolver{}`) pushed via `PushType` (NOT `PushTemplate`) with `std::conditional_t` to select the solver type
- Inner solver parameters: pass as `py::dict` in Python, extract with a `dp<T>(dict, key, default)` helper in the pybind `create_solver()` method — no hardcoded struct
- Template instantiation: 2 (execution) × N (inner solvers) files

### POMDP Solver API Pattern (e.g., RTDP-Bel)

POMDP solvers use two interfaces — observation-based (default) and belief-state:

**C++ algorithm:**
- `solve(vector<pair<State, double>> &initial_distribution)` — takes distribution, not single state
- Default interface: `get_best_action(Observation)`, `get_best_value(Observation)`, `is_solution_defined_for(Observation)` — internally tracks belief via Bayes rule
- Belief interface: `get_best_action_from_belief(Belief)`, `get_best_value_from_belief(Belief)`, `is_solution_defined_for_from_belief(Belief)`
- `reset_belief()` — resets tracked belief to initial
- Domain alias: `PythonDomainProxy<Texecution, SingleAgent, PartiallyObservable>`

**Python wrapper:**
- `_solve(from_memory=None)` overridden to call `domain.get_initial_state_distribution()` when None
- `_solve_from(initial_belief: Distribution[D.T_state])` — takes Distribution, not memory
- Domain D includes `UncertainInitialized` (provides `get_initial_state_distribution()`)
- Pybind converts Distribution → Belief via `get_values()` returning `list[tuple[T, float]]`
- `get_belief_policy()` returns dict with `frozenset` keys for hashable belief representation

### Functor Signature Differences Across Solvers

LRTDP's goal_checker/heuristic/callback functors include a `const std::size_t *` thread_id parameter. All other solvers (ILAOstar, LDFS, VI, PI) use simpler signatures without thread_id. When wrapping for meta-solvers:

```cpp
// Adapt simple functor to LRTDP's signature:
auto wrapped_gc = [sub_gc](Domain &d, const State &st, const std::size_t *) -> Predicate {
    return sub_gc(d, st);
};
```

### LP-Based Solver Pattern (IDUAL, Witness)

Some solvers (IDUAL for SSPs, Witness for POMDPs) solve linear programs via the HiGHS library. The architecture differs from search-based solvers:

**HiGHS API usage**: Build the LP incrementally using `addVar`/`addCol`/`addRow`, set objectives with `changeColCost`, modify coefficients with `changeCoeff`. All `addCol`/`addRow` calls append at the end — existing variable/constraint indices are stable.

**Incremental LP warm-starting**: When solving a series of growing LPs (e.g., IDUAL expands states each iteration), reuse a persistent `std::unique_ptr<Highs>` instance. Between iterations, update only changed coefficients and add new rows/columns. HiGHS automatically warm-starts from the previous simplex basis.

**Numerical stability fallback**: Warm-started simplex can fail on numerically difficult LPs (large coefficient ranges from dead-end costs vs transition probabilities). Use a three-tier fallback:
```cpp
_highs->run();
if (_highs->getModelStatus() != HighsModelStatus::kOptimal) {
    _highs->clearSolver();  // clear bad basis, retry from scratch
    _highs->run();
}
if (_highs->getModelStatus() != HighsModelStatus::kOptimal) {
    init_lp(s0);            // rebuild entire LP (last resort)
    _highs->run();
}
```

**Coefficient tracking**: `changeCoeff` sets absolute values, not deltas. When incrementally updating coefficients, maintain parallel tracking vectors (e.g., `_lp_col_obj[col]`, `_lp_col_c9[col]`) so you can compute the new absolute value after subtracting the old contribution.

**Reverse successor index**: For efficient incremental updates, maintain a reverse map `_lp_succ_to_cols[successor_state] → [(column_index, probability)]` so you can quickly find which LP columns reference a given state when that state transitions from fringe to expanded.

### Existing Solver Acronyms

`aostar`, `astar`, `bfws`, `despot`, `idual`, `ilaostar`, `iw`, `lrtdp`, `martdp`, `mcts`, `riw`, `sarsop`, `ssipp`, `vi`, `pi`, `ldfs`, `rtdp_bel`, `witness`.
