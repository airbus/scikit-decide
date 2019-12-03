/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "utils/python_hash_eq.hh"
#include "mcts.hh"
#include "core.hh"

namespace py = pybind11;

template <typename Texecution> struct GilControl;

template <>
struct GilControl<airlaps::SequentialExecution> {
    struct Acquire { Acquire() {} };
    struct Release { Release() {} };
};

template <>
struct GilControl<airlaps::ParallelExecution> {
    typedef py::gil_scoped_acquire Acquire;
    typedef py::gil_scoped_release Release;
};


template <typename Texecution>
class PyMCTSDomain {
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
                    return airlaps::python_hash(s._state);
                } catch(const py::error_already_set& e) {
                    throw std::runtime_error(e.what());
                }
            }
        };

        struct Equal {
            bool operator()(const State& s1, const State& s2) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    return airlaps::python_equal(s1._state, s2._state);
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
                throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python applicable action object for implementing get_elements()");
            }
            if (!py::hasattr(_applicable_actions, "sample")) {
                throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python applicable action object for implementing sample()");
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

        std::unique_ptr<Event> sample() const {
            typename GilControl<Texecution>::Acquire acquire;
            return std::make_unique<Event>(_applicable_actions.attr("sample")());
        }
    };

    struct NextStateDistribution {
        py::object _next_state_distribution;

        NextStateDistribution(const py::object& next_state_distribution)
        : _next_state_distribution(next_state_distribution) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_next_state_distribution, "get_values")) {
                throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python next state distribution object for implementing get_values()");
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

    struct TransitionOutcome {
        py::object _outcome;
        py::object _state;

        TransitionOutcome(const py::object& outcome)
        : _outcome(outcome) {
            typename GilControl<Texecution>::Acquire acquire;
            if (py::hasattr(_outcome, "state")) {
                _state = _outcome.attr("state");
            } else if (py::hasattr(_outcome, "observation")) {
                _state = _outcome.attr("observation");
            } else {
                throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python transition outcome object for providing 'state' or 'observation'");
            }
            if (!py::hasattr(_outcome, "value")) {
                throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python transition outcome object for providing 'value'");
            }
            if (!py::hasattr(_outcome, "termination")) {
                throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python transition outcome object for providing 'termination'");
            }
        }

        py::object state() {
            return _state;
        }

        double reward() {
            try {
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<double>(_outcome.attr("value").attr("reward"));
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's reward: ") + e.what());
                throw;
            }
        }

        bool terminal() {
            try {
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<bool>(_outcome.attr("termination"));
            } catch(const py::error_already_set& e) {
                spdlog::error(std::string("AIRLAPS exception when getting outcome's state: ") + e.what());
                throw;
            }
        }
    };

    PyMCTSDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!py::hasattr(domain, "get_next_state_distribution")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_next_state_distribution()");
        }
        if (!py::hasattr(domain, "sample")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: MCTS algorithm needs python domain for implementing is_terminal()");
        }
    }

    std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<ApplicableActionSpace>(_domain.attr("get_applicable_actions")(s._state));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    std::unique_ptr<NextStateDistribution> get_next_state_distribution(const State& s, const Event& a) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<NextStateDistribution>(_domain.attr("get_next_state_distribution")(s._state, a._event));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    std::unique_ptr<TransitionOutcome> sample(const State& s, const Event& e) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<TransitionOutcome>(_domain.attr("sample")(s._state, e._event));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when sampling from state ") + s.print() +
                          " with action " + e.print() + ": " + ex.what());
            throw;
        }
    }

    double get_transition_value(const State& s, const Event& a, const State& sp) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<double>(_domain.attr("get_transition_value")(s._state, a._event, sp._state).attr("reward"));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    bool is_terminal(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<bool>(_domain.attr("is_terminal")(s._state));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

private :
    py::object _domain;
};


class PyMCTSSolver {
public :
    PyMCTSSolver(py::object& domain,
                 std::size_t time_budget = 3600000,
                 std::size_t rollout_budget = 100000,
                 std::size_t max_depth = 1000,
                 double discount = 1.0,
                 bool uct_mode = true,
                 double ucb_constant = 1.0 / std::sqrt(2.0),
                 bool parallel = true,
                 bool debug_logs = false) {
        if (parallel) {
            if (uct_mode) {
                _implementation = std::make_unique<Implementation<airlaps::ParallelExecution>>(
                    domain, time_budget, rollout_budget, max_depth, discount, debug_logs
                );
                // _implementation->ucb_constant(ucb_constant);
            } else {
                spdlog::error("MCTS only supports MCTS at the moment.");
                throw std::runtime_error("MCTS only supports MCTS at the moment.");
            }
        } else {
            if (uct_mode) {
                _implementation = std::make_unique<Implementation<airlaps::SequentialExecution>>(
                    domain, time_budget, rollout_budget, max_depth, discount, debug_logs
                );
                // _implementation->ucb_constant(ucb_constant);
            } else {
                spdlog::error("MCTS only supports MCTS at the moment.");
                throw std::runtime_error("MCTS only supports MCTS at the moment.");
            }
        }
    }

    void clear() {
        _implementation->clear();
    }

    void solve(const py::object& s) {
        _implementation->solve(s);
    }

    py::bool_ is_solution_defined_for(const py::object& s) {
        return _implementation->is_solution_defined_for(s);
    }

    py::object get_next_action(const py::object& s) {
        return _implementation->get_next_action(s);
    }

    py::float_ get_utility(const py::object& s) {
        return _implementation->get_utility(s);
    }

    virtual py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    virtual py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
    }

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
        virtual void ucb_constant(double ucb_constant) =0;
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ get_nb_rollouts() =0;
    };

    template <typename Texecution>
    class Implementation : public BaseImplementation {
    public :
        Implementation(py::object& domain,
                       std::size_t time_budget = 3600000,
                       std::size_t rollout_budget = 100000,
                       std::size_t max_depth = 1000,
                       double discount = 1.0,
                       bool debug_logs = false) {

            _domain = std::make_unique<PyMCTSDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::MCTSSolver<PyMCTSDomain<Texecution>, Texecution>>(
                        *_domain,
                        time_budget,
                        rollout_budget,
                        max_depth,
                        discount,
                        debug_logs);
            _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                            py::module::import("sys").attr("stdout"));
            _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                            py::module::import("sys").attr("stderr"));
        }

        virtual void clear() {
            _solver->clear();
        }

        virtual void solve(const py::object& s) {
            typename GilControl<Texecution>::Release release;
            _solver->solve(s);
        }

        virtual py::bool_ is_solution_defined_for(const py::object& s) {
            return _solver->is_solution_defined_for(s);
        }

        virtual py::object get_next_action(const py::object& s) {
            return _solver->get_best_action(s).get();
        }

        virtual py::float_ get_utility(const py::object& s) {
            return _solver->get_best_value(s);
        }

        virtual py::int_ get_nb_of_explored_states() {
            return _solver->nb_of_explored_states();
        }

        virtual py::int_ get_nb_rollouts() {
            return _solver->nb_rollouts();
        }

        virtual void ucb_constant(double ucb_constant) {
            // _solver->action_selector().ucb_constant() = ucb_constant;
        }

    private :
        std::unique_ptr<PyMCTSDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::MCTSSolver<PyMCTSDomain<Texecution>, Texecution>> _solver;

        std::function<bool (const py::object&)> _goal_checker;
        std::function<double (const py::object&)> _heuristic;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pymcts(py::module& m) {
    py::class_<PyMCTSSolver> py_mcts_solver(m, "_MCTSSolver_");
        py_mcts_solver
            .def(py::init<py::object&,
                          std::size_t,
                          std::size_t,
                          std::size_t,
                          double,
                          bool,
                          double,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("discount")=1.0,
                 py::arg("uct_mode")=true,
                 py::arg("ucb_constant")=1.0/std::sqrt(2.0),
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyMCTSSolver::clear)
            .def("solve", &PyMCTSSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyMCTSSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyMCTSSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyMCTSSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyMCTSSolver::get_nb_of_explored_states)
            .def("get_nb_rollouts", &PyMCTSSolver::get_nb_rollouts)
        ;
}
