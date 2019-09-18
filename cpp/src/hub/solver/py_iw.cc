#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "iw.hh"
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
class PyIWDomain {
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
                    if (!py::hasattr(s._state, "__hash__") || s._state.attr("__hash__").is_none()) {
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python states for implementing __hash__");
                    }
                    // python __hash__ can return negative integers but c++ expects positive integers only
                    return s._state.attr("__hash__")().template cast<long>() + std::numeric_limits<long>::max();
                } catch(const py::error_already_set& e) {
                    spdlog::error(std::string("AIRLAPS exception when hashing states: ") + e.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const State& s1, const State& s2) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(s1._state, "__eq__") || s1._state.attr("__eq__").is_none()) {
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python states for implementing __eq__");
                    }
                    return s1._state.attr("__eq__")(s2._state).template cast<bool>();
                } catch(const py::error_already_set& e) {
                    spdlog::error(std::string("AIRLAPS exception when testing states equality: ") + e.what());
                    throw;
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
                    if (!py::hasattr(e._event, "__hash__") || e._event.attr("__hash__").is_none()) {
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python events for implementing __hash__");
                    }
                    // python __hash__ can return negative integers but c++ expects positive integers only
                    return e._event.attr("__hash__")().template cast<long>() + std::numeric_limits<long>::max();
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when hashing actions: ") + ex.what());
                    throw;
                }
            }
        };

        struct Equal {
            bool operator()(const Event& e1, const Event& e2) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    if (!py::hasattr(e1._event, "__eq__") || e1._event.attr("__eq__").is_none()) {
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python actions for implementing __eq__");
                    }
                    return e1._event.attr("__eq__")(e2._event).template cast<bool>();
                } catch(const py::error_already_set& ex) {
                    spdlog::error(std::string("AIRLAPS exception when testing actions equality: ") + ex.what());
                    throw;
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
                throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python applicable action object for implementing get_elements()");
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

    PyIWDomain(const py::object& domain)
    : _domain(domain) {
        if (!py::hasattr(domain, "wrapped_get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing wrapped_get_applicable_actions()");
        }
        if (!py::hasattr(domain, "wrapped_compute_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing wrapped_compute_sample()");
        }
        if (!py::hasattr(domain, "wrapped_get_next_state")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing wrapped_get_sample()");
        }
        if (!py::hasattr(domain, "get_transition_value")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing get_transition_value()");
        }
        if (!py::hasattr(domain, "is_goal")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing is_goal()");
        }
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing is_terminal()");
        }
    }

    std::unique_ptr<ApplicableActionSpace> get_applicable_actions(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return std::make_unique<ApplicableActionSpace>(_domain.attr("wrapped_get_applicable_actions")(s._state));
        } catch(const py::error_already_set& e) {
            spdlog::error(std::string("AIRLAPS exception when getting applicable actions in state ") + s.print() + ": " + e.what());
            throw;
        }
    }

    void compute_next_state(const State& s, const Event& e) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            _domain.attr("wrapped_compute_next_state")(s._state, e._event);
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when computing next state from state ") +
                          s.print() + " and applying action " + e.print() + ": " + ex.what());
            throw;
        }
    }

    py::object get_next_state(const State& s, const Event& e) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return _domain.attr("wrapped_get_next_state")(s._state, e._event);
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when getting next state from state ") +
                          s.print() + " and applying action " + e.print() + ": " + ex.what());
            throw;
        }
    }

    double get_transition_value(const State& s, const Event& e, const State& sp) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<double>(_domain.attr("get_transition_value")(s._state, e._event, sp._state).attr("cost"));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when getting value of transition (") +
                          s.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
            throw;
        }
    }

    bool is_goal(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<bool>(_domain.attr("is_goal")(s._state));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when testing goal condition of state ") +
                          s.print() + ": " + ex.what());
            throw;
        }
    }

    bool is_terminal(const State& s) {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            return py::cast<bool>(_domain.attr("is_terminal")(s._state));
        } catch(const py::error_already_set& ex) {
            spdlog::error(std::string("AIRLAPS exception when testing terminal condition of state ") +
                          s.print() + ": " + ex.what());
            throw;
        }
    }

private :
    py::object _domain;
};


class PyIWSolver {
public :

    PyIWSolver(py::object& domain,
                const unsigned int& nb_of_binary_features,
                const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>& state_binarizer,
                bool use_state_feature_hash = false,
                bool parallel = true,
                bool debug_logs = false) {
        if (parallel) {
            if (use_state_feature_hash) {
                _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::StateFeatureHash>>(
                    domain, nb_of_binary_features, state_binarizer, debug_logs
                );
            } else {
                _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::DomainStateHash>>(
                    domain, nb_of_binary_features, state_binarizer, debug_logs
                );
            }
        } else {
            if (use_state_feature_hash) {
                _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::StateFeatureHash>>(
                    domain, nb_of_binary_features, state_binarizer, debug_logs
                );
            } else {
                _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::DomainStateHash>>(
                    domain, nb_of_binary_features, state_binarizer, debug_logs
                );
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

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
    };

    template <typename Texecution, template <typename T> class Thashing_policy>
    class Implementation : public BaseImplementation {
    public :

        Implementation(py::object& domain,
                const unsigned int& nb_of_binary_features,
                const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>& state_binarizer,
                bool debug_logs = false)
            : _state_binarizer(state_binarizer) {
            _domain = std::make_unique<PyIWDomain<Texecution>>(domain);
            _solver = std::make_unique<airlaps::IWSolver<PyIWDomain<Texecution>, Thashing_policy, Texecution>>(
                                                                            *_domain,
                                                                            nb_of_binary_features,
                                                                            [this](const typename PyIWDomain<Texecution>::State& s,
                                                                                const std::function<void (const py::int_&)>& f)->void {
                                                                                typename GilControl<Texecution>::Acquire acquire;
                                                                                _state_binarizer(s._state, f);
                                                                            },
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
            try {
                return _solver->get_best_action(s).get();
            } catch (const std::runtime_error&) {
                return py::none();
            }
        }

        virtual py::float_ get_utility(const py::object& s) {
            try {
                return _solver->get_best_value(s);
            } catch (const std::runtime_error&) {
                return py::none();
            }
        }

    private :
        std::unique_ptr<PyIWDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::IWSolver<PyIWDomain<Texecution>, Thashing_policy, Texecution>> _solver;
        
        std::function<void (const py::object&, const std::function<void (const py::int_&)>&)> _state_binarizer;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pyiw(py::module& m) {
    py::class_<PyIWSolver> py_iw_solver(m, "_IWSolver_");
        py_iw_solver
            .def(py::init<py::object&,
                          const unsigned int&,
                          const std::function<void (const py::object&, const std::function<void (const py::int_&)>&)>&,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("nb_of_binary_features"),
                 py::arg("state_binarizer"),
                 py::arg("use_state_feature_hash"),
                 py::arg("parallel"),
                 py::arg("debug_logs")=false)
            .def("clear", &PyIWSolver::clear)
            .def("solve", &PyIWSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyIWSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyIWSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyIWSolver::get_utility, py::arg("state"))
        ;
}
