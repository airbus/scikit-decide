/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "riw.hh"
#include "core.hh"

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/python_domain_adapter.hh"

namespace py = pybind11;


template <typename Texecution>
class PyRIWDomain : public airlaps::PythonDomainAdapter<Texecution> {
public :
    
    PyRIWDomain(const py::object& domain, bool use_simulation_domain)
    : airlaps::PythonDomainAdapter<Texecution>(domain) {
        if (!py::hasattr(domain, "get_applicable_actions")) {
            throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python domain for implementing get_applicable_actions()");
        }
        if (!use_simulation_domain && !py::hasattr(domain, "reset")) {
            throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python domain for implementing reset() in environment mode");
        }
        if (!use_simulation_domain && !py::hasattr(domain, "step")) {
            throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python domain for implementing step() in environment mode");
        }
        if (use_simulation_domain && !py::hasattr(domain, "sample")) {
            throw std::invalid_argument("AIRLAPS exception: RIW algorithm needs python domain for implementing sample() in simulation mode");
        }
    }

};


template <typename Texecution>
using PyRIWFeatureVector = airlaps::PythonContainerAdapter<Texecution>;


class PyRIWSolver {
public :

    PyRIWSolver(py::object& domain,
                const std::function<py::object (py::object&, const py::object&)>& state_features,
                bool use_state_feature_hash = false,
                bool use_simulation_domain = false,
                unsigned int time_budget = 3600000,
                unsigned int rollout_budget = 100000,
                unsigned int max_depth = 1000,
                double exploration = 0.25,
                double discount = 1.0,
                bool online_node_garbage = false,
                bool parallel = true,
                bool debug_logs = false) {

        if (parallel) {
            if (use_state_feature_hash) {
                if (use_simulation_domain) {
                    _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::StateFeatureHash, airlaps::SimulationRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth, exploration, discount, online_node_garbage, debug_logs);
                } else {
                    _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::StateFeatureHash, airlaps::EnvironmentRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth, exploration, discount, online_node_garbage, debug_logs);
                }
            } else {
                if (use_simulation_domain) {
                    _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::DomainStateHash, airlaps::SimulationRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                } else {
                    _implementation = std::make_unique<Implementation<airlaps::ParallelExecution, airlaps::DomainStateHash, airlaps::EnvironmentRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                }
            }
        } else {
            if (use_state_feature_hash) {
                if (use_simulation_domain) {
                    _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::StateFeatureHash, airlaps::SimulationRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                } else {
                    _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::StateFeatureHash, airlaps::EnvironmentRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                }
            } else {
                if (use_simulation_domain) {
                    _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::DomainStateHash, airlaps::SimulationRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                } else {
                    _implementation = std::make_unique<Implementation<airlaps::SequentialExecution, airlaps::DomainStateHash, airlaps::EnvironmentRollout>>(
                        domain, state_features, time_budget, rollout_budget, max_depth,
                        exploration, discount, online_node_garbage, debug_logs);
                }
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

    py::int_ get_nb_of_explored_states() {
        return _implementation->get_nb_of_explored_states();
    }

    py::int_ get_nb_of_pruned_states() {
        return _implementation->get_nb_of_pruned_states();
    }

    py::int_ get_nb_rollouts() {
        return _implementation->get_nb_rollouts();
    }

    py::dict get_policy() {
        return _implementation->get_policy();
    }

    py::list get_action_prefix() {
        return _implementation->get_action_prefix();
    }

private :

    class BaseImplementation {
    public :
        virtual void clear() =0;
        virtual void solve(const py::object& s) =0;
        virtual py::bool_ is_solution_defined_for(const py::object& s) =0;
        virtual py::object get_next_action(const py::object& s) =0;
        virtual py::float_ get_utility(const py::object& s) =0;
        virtual py::int_ get_nb_of_explored_states() =0;
        virtual py::int_ get_nb_of_pruned_states() =0;
        virtual py::int_ get_nb_rollouts() =0;
        virtual py::dict get_policy() =0;
        virtual py::list get_action_prefix() =0;
    };

    template <typename Texecution,
              template <typename...> class Thashing_policy,
              template <typename...> class Trollout_policy>
    class Implementation : public BaseImplementation {
    public :

        Implementation(py::object& domain,
                       const std::function<py::object (py::object&, const py::object&)>& state_features,
                       unsigned int time_budget = 3600000,
                       unsigned int rollout_budget = 100000,
                       unsigned int max_depth = 1000,
                       double exploration = 0.25,
                       double discount = 1.0,
                       bool online_node_garbage = false,
                       bool debug_logs = false)
            : _state_features(state_features) {
            
            _domain = std::make_unique<PyRIWDomain<Texecution>>(domain,
                std::is_same<Trollout_policy<PyRIWDomain<Texecution>>, airlaps::SimulationRollout<PyRIWDomain<Texecution>>>::value);
            _solver = std::make_unique<airlaps::RIWSolver<PyRIWDomain<Texecution>, PyRIWFeatureVector<Texecution>, Thashing_policy, Trollout_policy, Texecution>>(
                                                                            *_domain,
                                                                            [this](PyRIWDomain<Texecution>& d, const typename PyRIWDomain<Texecution>::State& s)->std::unique_ptr<PyRIWFeatureVector<Texecution>> {
                                                                                try {
                                                                                    return std::make_unique<PyRIWFeatureVector<Texecution>>(d.call(_state_features, s._state));
                                                                                } catch (const py::error_already_set* e) {
                                                                                    typename airlaps::GilControl<Texecution>::Acquire acquire;
                                                                                    spdlog::error(std::string("AIRLAPS exception when calling state features: ") + e->what());
                                                                                    std::runtime_error err(e->what());
                                                                                    delete e;
                                                                                    throw err;
                                                                                }
                                                                            },
                                                                            time_budget,
                                                                            rollout_budget,
                                                                            max_depth,
                                                                            exploration,
                                                                            discount,
                                                                            online_node_garbage,
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
            typename airlaps::GilControl<Texecution>::Release release;
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

        virtual py::int_ get_nb_of_explored_states() {
            return _solver->get_nb_of_explored_states();
        }

        virtual py::int_ get_nb_of_pruned_states() {
            return _solver->get_nb_of_pruned_states();
        }

        virtual py::int_ get_nb_rollouts() {
            return _solver->get_nb_rollouts();
        }

        virtual py::dict get_policy() {
            py::dict d;
            auto&& p = _solver->policy();
            for (auto& e : p) {
                d[e.first._state] = py::make_tuple(e.second.first._event, e.second.second);
            }
            return d;
        }

        virtual py::list get_action_prefix() {
            py::list l;
            const auto& ll = _solver->action_prefix();
            for (const auto& e : ll) {
                l.append(e);
            }
            return l;
        }

    private :
        std::unique_ptr<PyRIWDomain<Texecution>> _domain;
        std::unique_ptr<airlaps::RIWSolver<PyRIWDomain<Texecution>, PyRIWFeatureVector<Texecution>, Thashing_policy, Trollout_policy, Texecution>> _solver;
        
        std::function<py::object (py::object&, const py::object&)> _state_features;

        std::unique_ptr<py::scoped_ostream_redirect> _stdout_redirect;
        std::unique_ptr<py::scoped_estream_redirect> _stderr_redirect;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


void init_pyriw(py::module& m) {
    py::class_<PyRIWSolver> py_riw_solver(m, "_RIWSolver_");
        py_riw_solver
            .def(py::init<py::object&,
                          const std::function<py::object (py::object&, const py::object&)>&,
                          bool,
                          bool,
                          unsigned int,
                          unsigned int,
                          unsigned int,
                          double,
                          double,
                          bool,
                          bool,
                          bool>(),
                 py::arg("domain"),
                 py::arg("state_features"),
                 py::arg("use_state_feature_hash")=false,
                 py::arg("use_simulation_domain")=false,
                 py::arg("time_budget")=3600000,
                 py::arg("rollout_budget")=100000,
                 py::arg("max_depth")=1000,
                 py::arg("exploration")=0.25,
                 py::arg("discount")=1.0,
                 py::arg("online_node_garbage")=false,
                 py::arg("parallel")=true,
                 py::arg("debug_logs")=false)
            .def("clear", &PyRIWSolver::clear)
            .def("solve", &PyRIWSolver::solve, py::arg("state"))
            .def("is_solution_defined_for", &PyRIWSolver::is_solution_defined_for, py::arg("state"))
            .def("get_next_action", &PyRIWSolver::get_next_action, py::arg("state"))
            .def("get_utility", &PyRIWSolver::get_utility, py::arg("state"))
            .def("get_nb_of_explored_states", &PyRIWSolver::get_nb_of_explored_states)
            .def("get_nb_of_pruned_states", &PyRIWSolver::get_nb_of_pruned_states)
            .def("get_nb_rollouts", &PyRIWSolver::get_nb_rollouts)
            .def("get_policy", &PyRIWSolver::get_policy)
            .def("get_action_prefix", &PyRIWSolver::get_action_prefix)
        ;
}
