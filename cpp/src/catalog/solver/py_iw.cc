#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "iw.hh"
#include "gym_spaces.hh"
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
                    if (!py::hasattr(s._state, "__hash__")) {
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python states for implementing __hash__()");
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
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python states for implementing __eq__()");
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
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python events for implementing __hash__()");
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
                        throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python actions for implementing __eq__()");
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
        if (!py::hasattr(domain, "is_terminal")) {
            throw std::invalid_argument("AIRLAPS exception: IW algorithm needs python domain for implementing is_terminal()");
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
            return py::cast<double>(_domain.attr("get_transition_value")(s._state, e._event, sp._state).attr("reward"));
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


template <typename Texecution>
class PyIWSolver {
public :
    PyIWSolver(py::object& domain,
               const std::string& planner = "bfs",
               const std::function<py::array_t<int> (const py::object&)>& state_to_feature_atoms = [](const py::object& s)-> py::array_t<int> {return py::array_t<int>();},
               const std::string& default_encoding_type = "byte",
               double default_encoding_space_relative_precision = 0.001,
               size_t frameskip = 15,
               int simulator_budget = 150000,
               double time_budget = std::numeric_limits<double>::infinity(),
               bool novelty_subtables = false,
               bool random_actions = false,
               size_t max_rep = 30,
               int nodes_threshold = 50000,
               size_t max_depth = 1500,
               bool break_ties_using_rewards = false,
               double discount = 1.0,
               bool debug_logs = false) {
        _domain = std::make_unique<PyIWDomain<Texecution>>(domain);

        // Are we using the default state_to_feature_atoms?
        std::function<std::vector<int> (const typename PyIWDomain<Texecution>::State&)> encoder;
        try {
            if (state_to_feature_atoms(py::object()).size() == 0) {
                if (default_encoding_type == "byte") {
                    _gym_observation_space = airlaps::GymSpace::import_from_python(
                                                domain.attr("observation_space"),
                                                airlaps::GymSpace::ENCODING_BYTE_VECTOR,
                                                default_encoding_space_relative_precision);
                    _gym_action_space = airlaps::GymSpace::import_from_python(
                                                domain.attr("action_space"),
                                                airlaps::GymSpace::ENCODING_BYTE_VECTOR,
                                                default_encoding_space_relative_precision);
                    
                } else if (default_encoding_type == "variable") {
                    _gym_observation_space = airlaps::GymSpace::import_from_python(
                                                domain.attr("observation_space"),
                                                airlaps::GymSpace::ENCODING_VARIABLE_VECTOR,
                                                default_encoding_space_relative_precision);
                    _gym_action_space = airlaps::GymSpace::import_from_python(
                                                domain.attr("action_space"),
                                                airlaps::GymSpace::ENCODING_VARIABLE_VECTOR,
                                                default_encoding_space_relative_precision);
                } else {
                    py::print("ERROR: unsupported feature atom vector encoding '" + default_encoding_type + "'");
                }
                _gym_space_relative_precision = default_encoding_space_relative_precision;
                encoder = [this](const typename PyIWDomain<Texecution>::State& s) -> std::vector<int> {
                    typename GilControl<Texecution>::Acquire acquire;
                    return _gym_observation_space->convert_element_to_feature_atoms(s._state);
                };
            }
        } catch (...) {
            // We are using an encoder provided by the user
            encoder = [&state_to_feature_atoms](const typename PyIWDomain<Texecution>::State& s) -> std::vector<int> {
                typename GilControl<Texecution>::Acquire acquire;
                return py::cast<std::vector<int>>(state_to_feature_atoms(s._state));
            };
        }

        if (planner == "bfs") {
            _solver = std::make_unique<airlaps::BfsIW<PyIWDomain<Texecution>, Texecution>>(
                        *_domain,
                        encoder,
                        frameskip,
                        simulator_budget,
                        time_budget,
                        novelty_subtables,
                        random_actions,
                        max_rep,
                        nodes_threshold,
                        break_ties_using_rewards,
                        discount,
                        debug_logs);
        } else if (planner == "rollout") {
            _solver = std::make_unique<airlaps::RolloutIW<PyIWDomain<Texecution>, Texecution>>(
                        *_domain,
                        encoder,
                        frameskip,
                        simulator_budget,
                        time_budget,
                        novelty_subtables,
                        random_actions,
                        max_rep,
                        nodes_threshold,
                        max_depth,
                        discount,
                        debug_logs);
        } else {
            py::print("ERROR: unsupported IW planner '" + planner + "'");
        }
        
        _stdout_redirect = std::make_unique<py::scoped_ostream_redirect>(std::cout,
                                                                         py::module::import("sys").attr("stdout"));
        _stderr_redirect = std::make_unique<py::scoped_estream_redirect>(std::cerr,
                                                                         py::module::import("sys").attr("stderr"));
    }

    void reset() {
        _solver->reset();
    }

    void solve(const py::object& s) {
        typename GilControl<Texecution>::Release release;
        _solver->solve(s);
    }

    py::object get_next_action(const py::object& s) {
        return _solver->get_best_action(s).get();
    }

    py::float_ get_utility(const py::object& s) {
        return _solver->get_best_value(s);
    }

private :
    std::unique_ptr<PyIWDomain<Texecution>> _domain;
    std::unique_ptr<airlaps::IWSolver<PyIWDomain<Texecution>, Texecution>> _solver;
    std::unique_ptr<airlaps::GymSpace> _gym_observation_space;
    std::unique_ptr<airlaps::GymSpace> _gym_action_space;
    double _gym_space_relative_precision;

    std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
    std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
};


void init_pyiw(py::module& m) {
    py::class_<PyIWSolver<airlaps::SequentialExecution>> py_iw_seq_solver(m, "_IWSeqSolver_");
        py_iw_seq_solver
            .def(py::init<py::object&,
                          std::string,
                          const std::function<py::array_t<int> (const py::object&)>&,
                          std::string,
                          size_t,
                          int,
                          double,
                          bool,
                          bool,
                          size_t,
                          int,
                          size_t,
                          bool,
                          double,
                          bool>(),
                 py::arg("domain"),
                 py::arg("planner")="bfs",
                 py::arg("state_to_feature_atoms_encoder")=[](const py::object& s)-> py::array_t<int> {return py::array_t<int>();},
                 py::arg("default_encoding_type"),
                 py::arg("frameskip")=15,
                 py::arg("simulator_budget")=150000,
                 py::arg("time_budget")=std::numeric_limits<double>::infinity(),
                 py::arg("novelty_subtables")=false,
                 py::arg("random_actions")=false,
                 py::arg("max_rep")=30,
                 py::arg("nodes_threshold")=50000,
                 py::arg("max_depth")=1500,
                 py::arg("break_ties_using_rewards")=false,
                 py::arg("discount")=1.0,
                 py::arg("debug_logs")=false)
            .def("reset", &PyIWSolver<airlaps::SequentialExecution>::reset)
            .def("solve", &PyIWSolver<airlaps::SequentialExecution>::solve, py::arg("state"))
            .def("get_next_action", &PyIWSolver<airlaps::SequentialExecution>::get_next_action, py::arg("state"))
            .def("get_utility", &PyIWSolver<airlaps::SequentialExecution>::get_utility, py::arg("state"))
        ;
    
    py::class_<PyIWSolver<airlaps::ParallelExecution>> py_iw_par_solver(m, "_IWParSolver_");
        py_iw_par_solver
            .def(py::init<py::object&,
                          std::string,
                          const std::function<py::array_t<int> (const py::object&)>&,
                          std::string,
                          size_t,
                          int,
                          double,
                          bool,
                          bool,
                          size_t,
                          int,
                          size_t,
                          bool,
                          double,
                          bool>(),
                 py::arg("domain"),
                 py::arg("planner")="bfs",
                 py::arg("state_to_feature_atoms_encoder")=[](const py::object& s)-> py::array_t<int> {return py::array_t<int>();},
                 py::arg("deefault_encooding_type"),
                 py::arg("frameskip")=15,
                 py::arg("simulator_budget")=150000,
                 py::arg("time_budget")=std::numeric_limits<double>::infinity(),
                 py::arg("novelty_subtables")=false,
                 py::arg("random_actions")=false,
                 py::arg("max_rep")=30,
                 py::arg("nodes_threshold")=50000,
                 py::arg("max_depth")=1500,
                 py::arg("break_ties_using_rewards")=false,
                 py::arg("discount")=1.0,
                 py::arg("debug_logs")=false)
            .def("reset", &PyIWSolver<airlaps::ParallelExecution>::reset)
            .def("solve", &PyIWSolver<airlaps::ParallelExecution>::solve, py::arg("state"))
            .def("get_next_action", &PyIWSolver<airlaps::ParallelExecution>::get_next_action, py::arg("state"))
            .def("get_utility", &PyIWSolver<airlaps::ParallelExecution>::get_utility, py::arg("state"))
        ;
}