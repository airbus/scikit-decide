#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "aostar.hh"
#include "core.hh"

namespace py = pybind11;

class PyAOStarDomain {
public :
    struct State {
        py::object _state;

        State() {}
        State(const py::object& s) : _state(s) {}

        std::string print() const { return py::str(_state); }

        struct Hash {
            std::size_t operator()(const State& s) const {
                if (!py::hasattr(s._state, "__hash__")) {
                    throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python states for implementing __hash__()");
                }
                // python __hash__ can return negative integers but c++ expects positive integers only
                return s._state.attr("__hash__")().cast<long>() + std::numeric_limits<long>::max();
            }
        };

        struct Equal {
            bool operator()(const State& s1, const State& s2) const {
                if (!py::hasattr(s1._state, "__eq__")) {
                    throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python states for implementing __eq__()");
                }
                return s1._state.attr("__eq__")(s2._state).cast<bool>();
            }
        };
    };

    struct Event {
        py::object _event;

        Event() {}
        Event(const py::object& e) : _event(e) {}
        Event(const py::handle& e) : _event(py::reinterpret_borrow<py::object>(e)) {}
        
        const py::object& get() const { return _event; }
        std::string print() const { return py::str(_event); }
    };

    struct ApplicableActionSpace { // don't inherit from airlaps::EnumerableSpace since otherwise we would need to copy the applicable action python object into a c++ iterable object
        py::object _applicable_actions;

        ApplicableActionSpace(const py::object& applicable_actions)
        : _applicable_actions(applicable_actions) {
            if (!py::hasattr(_applicable_actions, "get_elements")) {
                throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python applicable action object for implementing get_elements()");
            }
        }

        py::object get_elements() const {
            return _applicable_actions.attr("get_elements")();
        }
    };

    struct NextStateDistribution {
        py::object _next_state_distribution;

        NextStateDistribution(const py::object& next_state_distribution)
        : _next_state_distribution(next_state_distribution) {
            if (!py::hasattr(_next_state_distribution, "get_values")) {
                throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python next state distribution object for implementing get_values()");
            }
        }

        py::object get_values() const {
            return _next_state_distribution.attr("get_values")();
        }
    };

    struct OutcomeExtractor {
        py::object _state;
        double _probability;

        OutcomeExtractor(const py::handle& o) {
            if (!py::isinstance<py::tuple>(o)) {
                throw std::invalid_argument("AIRLAPS exception: python next state distribution returned value should be an iterable over tuple objects");
            }
            py::tuple t = o.cast<py::tuple>();
            _state = t[0];
            _probability = t[1].cast<double>();
        }
        const py::object& state() const { return _state; }
        const double& probability() const { return _probability; }
    };

    PyAOStarDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: AO* algorithm needs python domain for implementing get_transition_value()");
        }
    }

    std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
        return std::make_unique<ApplicableActionSpace>(_domain.attr("get_applicable_actions")(s._state));
    }

    std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const py::handle& a) {
        return std::make_unique<NextStateDistribution>(_domain.attr("get_next_state_distribution")(s._state, a));
    }

    double get_transition_value(const State& s, const py::handle& a, const State& sp) {
        return _domain.attr("get_transition_value")(s._state, a, sp._state).attr("cost").cast<double>();
    }

private :
    py::object _domain;
};


class PyAOStarSolver {
public :
    PyAOStarSolver(py::object& domain,
                   const std::function<bool (const py::object&)>& goal_checker,
                   const std::function<double (const py::object&)>& heuristic,
                   double discount = 1.0,
                   unsigned int max_tip_expansions = 1,
                   bool detect_cycles = false)
        : _goal_checker(goal_checker), _heuristic(heuristic) {
        _domain = std::make_unique<PyAOStarDomain>(domain);
        _solver = std::make_unique<airlaps::AOStarSolver<PyAOStarDomain>>(*_domain,
                                                                          [this](const PyAOStarDomain::State& s)->bool {return _goal_checker(s._state);},
                                                                          [this](const PyAOStarDomain::State& s)->double {return _heuristic(s._state);},
                                                                          discount,
                                                                          max_tip_expansions,
                                                                          detect_cycles);
    }

    void reset() {
        _solver->reset();
    }

    void solve(const py::object& s) {
        _solver->solve(s);
    }

    py::object get_next_action(const py::object& s) {
        return _solver->get_best_action(s).get();
    }

    py::float_ get_utility(const py::object& s) {
        return _solver->get_best_value(s);
    }

private :
    std::unique_ptr<PyAOStarDomain> _domain;
    std::unique_ptr<airlaps::AOStarSolver<PyAOStarDomain>> _solver;

    std::function<bool (const py::object&)> _goal_checker;
    std::function<double (const py::object&)> _heuristic;
};

void init_pyaostar(py::module& m) {
    py::class_<PyAOStarSolver> py_aostar_solver(m, "__AOStarSolver");
        py_aostar_solver
            .def(py::init<py::object&,
                          const std::function<bool (const py::object&)>&,
                          const std::function<double (const py::object&)>&,
                          double,
                          unsigned int,
                          bool>(),
                 py::arg("domain"),
                 py::arg("goal_checker"),
                 py::arg("heuristic"),
                 py::arg("discount")=1.0,
                 py::arg("max_tip_expansions")=1,
                 py::arg("detect_cycles")=false)
            .def("reset", &PyAOStarSolver::reset)
            .def("solve", &PyAOStarSolver::solve, py::arg("state"))
            .def("get_next_action", &PyAOStarSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyAOStarSolver::get_utility, py::arg("state"))
        ;
}