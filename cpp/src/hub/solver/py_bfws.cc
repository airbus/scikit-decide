#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "bfws.hh"
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
class PyBFWSDomain {
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
                        throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python states for implementing __hash__()");
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
                        throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python states for implementing __eq__()");
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

        struct Hash {
            std::size_t operator()(const Event& e) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(e._event, "__hash__")) {
                        throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python events for implementing __hash__()");
                    }
                    // python __hash__ can return negative integers but c++ expects positive integers only
                    return e._event.attr("__hash__")().template cast<long>() + std::numeric_limits<long>::max();
                } catch(const py::error_already_set& ex) {
                    throw std::runtime_error(ex.what());
                }
            }
        };

        struct Equal {
            bool operator()(const Event& e1, const Event& e2) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(e1._event, "__eq__")) {
                        throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python actions for implementing __eq__()");
                    }
                    return e1._event.attr("__eq__")(e2._event).template cast<bool>();
                } catch(const py::error_already_set& ex) {
                    throw std::runtime_error(ex.what());
                }
            }
        };
    };

    struct ApplicableActionSpace { // don't inherit from airlaps::EnumerableSpace since otherwise we would need to copy the applicable action python object into a c++ iterable object
        py::object _applicable_actions;

        ApplicableActionSpace(const py::object& applicable_actions)
        : _applicable_actions(applicable_actions) {
            typename GilControl<Texecution>::Acquire acquire;
            if (!py::hasattr(_applicable_actions, "get_elements")) {
                throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python applicable action object for implementing get_elements()");
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

    PyBFWSDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "wrapped_get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing wrapped_get_applicable_actions()");
        }
        if (!py::hasattr(domain, "wrapped_compute_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing wrapped_compute_sample()");
        }
        if (!py::hasattr(domain, "wrapped_get_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing wrapped_get_sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: BFWS algorithm needs python domain for implementing is_terminal()");
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

    void compute_next_state(const State& s, const Event& e) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            _domain.attr("wrapped_compute_next_state")(s._state, e._event);
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    py::object get_next_state(const State& s, const Event& e) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return _domain.attr("wrapped_get_next_state")(s._state, e._event);
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    double get_transition_value(const State& s, const Event& e, const State& sp) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<double>(_domain.attr("get_transition_value")(s._state, e._event, sp._state).attr("cost"));
        } catch(const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

private :
    py::object _domain;
};


template <typename Texecution>
class PyBFWSSolver {
public :
    PyBFWSSolver(py::object& domain,
                 const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>& state_binarizer,
                 const std::function<double (const py::object&)>& heuristic,
                 const std::function<bool (const py::object&)>& termination_checker,
                 bool debug_logs = false)
        : _state_binarizer(state_binarizer), _heuristic(heuristic), _termination_checker(termination_checker) {
          _domain = std::make_unique<PyBFWSDomain<Texecution>>(domain);
          _solver = std::make_unique<airlaps::BFWSSolver<PyBFWSDomain<Texecution>, Texecution>>(
                                                                        *_domain,
                                                                        [this](const typename PyBFWSDomain<Texecution>::State& s,
                                                                               const std::function<void (const py::int_&)>& f)->void {
                                                                            typename GilControl<Texecution>::Acquire acquire;
                                                                            _state_binarizer(s._state, f);
                                                                        },
                                                                        [this](const typename PyBFWSDomain<Texecution>::State& s)->double {
                                                                            typename GilControl<Texecution>::Acquire acquire;
                                                                            return _heuristic(s._state);
                                                                        },
                                                                        [this](const typename PyBFWSDomain<Texecution>::State& s)->bool {
                                                                            typename GilControl<Texecution>::Acquire acquire;
                                                                            return _termination_checker(s._state);
                                                                        },
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
    std::unique_ptr<PyBFWSDomain<Texecution>> _domain;
    std::unique_ptr<airlaps::BFWSSolver<PyBFWSDomain<Texecution>, Texecution>> _solver;
    
    std::function<void (const py::object&, const std::function<void (const py::int_&)>&)> _state_binarizer;
    std::function<double (const py::object&)> _heuristic;
    std::function<bool (const py::object&)> _termination_checker;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
};


template <typename Texecution>
void declare_bfws_solver(py::module& m, const char* name) {
    py::class_<PyBFWSSolver<Texecution>> py_bfws_solver(m, name);
        py_bfws_solver
            .def(py::init<py::object&,
                          const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>&,
                          const std::function<double (const py::object&)>&,
                          const std::function<bool (const py::object&)>&,
                          bool>(),
                 py::arg("domain"),
                 py::arg("state_binarizer"),
                 py::arg("heuristic"),
                 py::arg("termination_checker"),
                 py::arg("debug_logs")=false)
            .def("clear", &PyBFWSSolver<Texecution>::clear)
            .def("solve", &PyBFWSSolver<Texecution>::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyBFWSSolver<Texecution>::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyBFWSSolver<Texecution>::get_next_action, py::arg("state"))
            .def("get_utility", &PyBFWSSolver<Texecution>::get_utility, py::arg("state"))
        ;
}


void init_pybfws(py::module& m) {
    declare_bfws_solver<airlaps::SequentialExecution>(m, "_BFWSSeqSolver_");
    declare_bfws_solver<airlaps::ParallelExecution>(m, "_BFWSParSolver_");
}
