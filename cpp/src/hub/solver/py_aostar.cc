#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "aostar.hh"
#include "core.hh"

namespace py = pybind11;

template <typename Texecution> struct GilControl;

template <>
struct GilControl<airlaps::SequentialExecution> {
    struct Acquire {};
    struct Release {};
};

template <>
struct GilControl<airlaps::ParallelExecution> {
    typedef py::gil_scoped_acquire Acquire;
    typedef py::gil_scoped_release Release;
};


template <typename Texecution>
class PyAOStarDomain {
public :
    struct State {
        py::object _state;

        State() {}
        State(const py::object& s) : _state(s) {}

        ~State() {
            typename GilControl<Texecution>::Acquire acquire;
            _state = py::object();
        }

        std::string print() const {
            typename GilControl<Texecution>::Acquire acquire;
            return py::str(_state);
        }

        struct Hash {
            std::size_t operator()(const State& s) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(s._state, "__hash__")) {
                        throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python states for implementing __hash__()");
                    }
                    // python __hash__ can return negative integers but c++ expects positive integers only
                    return s._state.attr("__hash__")().template cast<long>() + std::numeric_limits<long>::max();
                } catch(const py::error_already_set& e) {
                    throw std::runtime_error(e.what());
                }
            }
        };

        struct Equal {
            bool operator()(const State& s1, const State& s2) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(s1._state, "__eq__")) {
                        throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python states for implementing __eq__()");
                    }
                    return s1._state.attr("__eq__")(s2._state).template cast<bool>();
                } catch(const py::error_already_set& e) {
                    throw std::runtime_error(e.what());
                }
            }
        };
    };

    struct Event {
        py::object _event;

        Event() {}
        Event(const py::object& e) : _event(e) {}
        Event(const py::handle& e) : _event(py::reinterpret_borrow<py::object>(e)) {}

        ~Event() {
            typename GilControl<Texecution>::Acquire acquire;
            _event = py::object();
        }
        
        const py::object& get() const { return _event; }

        std::string print() const {
            typename GilControl<Texecution>::Acquire acquire;
            return py::str(_event);
        }
    };

    struct ApplicableActionSpace { // don't inherit from airlaps::EnumerableSpace since otherwise we would need to copy the applicable action python object into a c++ iterable object
        py::object _applicable_actions;

        ApplicableActionSpace(const py::object& applicable_actions)
        : _applicable_actions(applicable_actions) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_applicable_actions, "get_elements")) {
                throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python applicable action object for implementing get_elements()");
            }
        }

        ~ApplicableActionSpace() {
            typename GilControl<Texecution>::Acquire acquire;
            _applicable_actions = py::object();
        }

        struct ApplicableActionSpaceElements {
            py::object _elements;
            
            ApplicableActionSpaceElements(const py::object& elements)
            : _elements(elements) {}

            ~ApplicableActionSpaceElements() {
                typename GilControl<Texecution>::Acquire acquire;
                _elements = py::object();
            }

            py::iterator begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _elements.begin();
            }

            py::iterator end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _elements.end();
            }
        };

        ApplicableActionSpaceElements get_elements() const {
            typename GilControl<Texecution>::Acquire acquire;
            return ApplicableActionSpaceElements(_applicable_actions.attr("get_elements")());
        }
    };

    struct NextStateDistribution {
        py::object _next_state_distribution;

        NextStateDistribution(const py::object& next_state_distribution)
        : _next_state_distribution(next_state_distribution) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_next_state_distribution, "get_values")) {
                throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python next state distribution object for implementing get_values()");
            }
        }

        ~NextStateDistribution() {
            typename GilControl<Texecution>::Acquire acquire;
            _next_state_distribution = py::object();
        }

        struct NextStateDistributionValues {
            py::object _values;

            NextStateDistributionValues(const py::object& values)
            : _values(values) {}

            ~NextStateDistributionValues() {
                typename GilControl<Texecution>::Acquire acquire;
                _values = py::object();
            }

            py::iterator begin() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _values.begin();
            }

            py::iterator end() const {
                typename GilControl<Texecution>::Acquire acquire;
                return _values.end();
            }
        };

        NextStateDistributionValues get_values() const {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                return NextStateDistributionValues(_next_state_distribution.attr("get_values")());
            } catch(const py::error_already_set& e) {
                throw std::runtime_error(e.what());
            }
        }
    };

    struct OutcomeExtractor {
        py::object _state;
        double _probability;

        OutcomeExtractor(const py::handle& o) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::isinstance<py::tuple>(o)) {
                throw std::invalid_argument("AIRLAPS exception: python next state distribution returned value should be an iterable over tuple objects");
            }
            py::tuple t = o.cast<py::tuple>();
            _state = t[0];
            _probability = t[1].cast<double>();
        }

        ~OutcomeExtractor() {
            typename GilControl<Texecution>::Acquire acquire;
            _state = py::object();
        }

        const py::object& state() const { return _state; }
        const double& probability() const { return _probability; }
    };

    PyAOStarDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "wrapped_get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing wrapped_get_applicable_actions()");
        }
        if (!py::hasattr(domain, "wrapped_compute_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing wrapped_compute_next_state_distribution()");
        }
        if (!py::hasattr(domain, "wrapped_get_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing wrapped_get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing get_transition_value()");
        }
    }

    std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<ApplicableActionSpace>(_domain.attr("wrapped_get_applicable_actions")(s._state));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    void compute_next_state_distribution(const State& s, const py::handle& a) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            _domain.attr("wrapped_compute_next_state_distribution")(s._state, a);
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const py::handle& a) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<NextStateDistribution>(_domain.attr("wrapped_get_next_state_distribution")(s._state, a));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    double get_transition_value(const State& s, const py::handle& a, const State& sp) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<double>(_domain.attr("get_transition_value")(s._state, a, sp._state).attr("cost"));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

private :
    py::object _domain;
};


template <typename Texecution>
class PyAOStarSolver {
public :
    PyAOStarSolver(py::object& domain,
                   const std::function<bool (const py::object&)>& goal_checker,
                   const std::function<double (const py::object&)>& heuristic,
                   double discount = 1.0,
                   unsigned int max_tip_expansions = 1,
                   bool detect_cycles = false,
                   bool debug_logs = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
        _domain = std::make_unique<PyAOStarDomain<Texecution>>(domain);
        _solver = std::make_unique<airlaps::AOStarSolver<PyAOStarDomain<Texecution>, Texecution>>(
                                                                        *_domain,
                                                                        [this](const typename PyAOStarDomain<Texecution>::State& s)->bool {
                                                                            typename GilControl<Texecution>::Acquire acquire;
                                                                            return _goal_checker(s._state);
                                                                        },
                                                                        [this](const typename PyAOStarDomain<Texecution>::State& s)->double {
                                                                            typename GilControl<Texecution>::Acquire acquire;
                                                                            return _heuristic(s._state);
                                                                        },
                                                                        discount,
                                                                        max_tip_expansions,
                                                                        detect_cycles,
                                                                        debug_logs);
        _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                         py::module::import("sys").attr("stdout"));
        _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                         py::module::import("sys").attr("stderr"));
    }

    void clear() {
        _solver->clear();
    }

    void solve(const py::object& s) {
        typename GilControl<Texecution>::Release release;
        _solver->solve(s);
    }

    py::bool_ is_solution_defined_for(const py::object& s) {
        return _solver->is_solution_defined_for(s);
    }

    py::object get_next_action(const py::object& s) {
        return _solver->get_best_action(s).get();
    }

    py::float_ get_utility(const py::object& s) {
        return _solver->get_best_value(s);
    }

private :
    std::unique_ptr<PyAOStarDomain<Texecution>> _domain;
    std::unique_ptr<airlaps::AOStarSolver<PyAOStarDomain<Texecution>, Texecution>> _solver;

    std::function<bool (const py::object&)> _goal_checker;
    std::function<double (const py::object&)> _heuristic;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
};


template <typename Texecution>
void declare_aostar_solver(py::module& m, const char* name) {
    py::class_<PyAOStarSolver<Texecution>> py_aostar_seq_solver(m, name);
        py_aostar_seq_solver
            .def(py::init<py::object&,
                          const std::function<bool (const py::object&)>&,
                          const std::function<double (const py::object&)>&,
                          double,
                          unsigned int,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("discount")=1.0,
                 py::arg("max_tip_expansions")=1,
                 py::arg("detect_cycles")=false,
                 py::arg("debug_logs")=false)
            .def("clear", &PyAOStarSolver<Texecution>::clear)
            .def("solve", &PyAOStarSolver<Texecution>::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyAOStarSolver<Texecution>::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyAOStarSolver<Texecution>::get_next_action, py::arg("state"))
            .def("get_utility", &PyAOStarSolver<Texecution>::get_utility, py::arg("state"))
        ;
}


void init_pyaostar(py::module& m) {
    declare_aostar_solver<airlaps::SequentialExecution>(m, "_AOStarSeqSolver_");
    declare_aostar_solver<airlaps::ParallelExecution>(m, "_AOStarParSolver_");
}