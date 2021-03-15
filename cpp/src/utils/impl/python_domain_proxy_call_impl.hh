/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_CALL_IMPL_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_CALL_IMPL_HH

#include <nngpp/nngpp.h>
#include <nngpp/protocol/pull0.h>

#include "utils/python_gil_control.hh"
#include "utils/python_globals.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === PythonDomainProxy::Implementation<SequentialExecution> implementation ===

template <typename Texecution, typename Tagent, typename Tobservability, typename Tcontrollability, typename Tmemory>
template <typename TexecutionPolicy>
struct PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Implementation<
                TexecutionPolicy,
                typename std::enable_if<std::is_same<TexecutionPolicy, SequentialExecution>::value>::type> {
    
    std::unique_ptr<py::object> _domain;
    
    Implementation(const py::object& domain);
    ~Implementation();

    void close();
    std::size_t get_parallel_capacity();
    ApplicableActionSpace get_applicable_actions(const Memory& m, const std::size_t* thread_id = nullptr);

    template <typename TTagent = Tagent,
              typename TTaction = Action,
              typename TagentApplicableActions = typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace>
    std::enable_if_t<std::is_same<TTagent, MultiAgent>::value, TagentApplicableActions>
    get_agent_applicable_actions(const Memory& m,
                                 const TTaction& other_agents_actions,
                                 const Agent& agent,
                                 const std::size_t* thread_id = nullptr);

    Observation reset(const std::size_t* thread_id = nullptr);
    EnvironmentOutcome step(const Event& e, const std::size_t* thread_id = nullptr);
    EnvironmentOutcome sample(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    State get_next_state(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    Value get_transition_value(const Memory& m, const Event& e, const State& sp, const std::size_t* thread_id = nullptr);
    bool is_goal(const State& s, const std::size_t* thread_id = nullptr);
    bool is_terminal(const State& s, const std::size_t* thread_id = nullptr);

    template <typename Tfunction, typename ... Types>
    std::unique_ptr<py::object> call([[maybe_unused]] const std::size_t* thread_id,
                                     const Tfunction& func,
                                     const Types& ... args) {
        try {
            return std::make_unique<py::object>(func(*_domain, args..., py::none()));
        } catch(const py::error_already_set* e) {
            std::runtime_error err(e->what());
            delete e;
            throw err;
        }
    }
};

// === PythonDomainProxy::Implementation<ParallelExecution> implementation ===

template <typename Texecution, typename Tagent, typename Tobservability, typename Tcontrollability, typename Tmemory>
template <typename TexecutionPolicy>
struct PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::Implementation<
                TexecutionPolicy,
                typename std::enable_if<std::is_same<TexecutionPolicy, ParallelExecution>::value>::type> {
    
    std::unique_ptr<py::object> _domain;
    std::vector<std::unique_ptr<nng::socket>> _connections;
    
    Implementation(const py::object& domain);
    ~Implementation();

    void close();
    std::size_t get_parallel_capacity();
    ApplicableActionSpace get_applicable_actions(const Memory& m, const std::size_t* thread_id = nullptr);

    template <typename TTagent = Tagent,
              typename TTaction = Action,
              typename TagentApplicableActions = typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace>
    std::enable_if_t<std::is_same<TTagent, MultiAgent>::value, TagentApplicableActions>
    get_agent_applicable_actions(const Memory& m,
                                 const TTaction& other_agents_actions,
                                 const Agent& agent,
                                 const std::size_t* thread_id = nullptr);

    Observation reset(const std::size_t* thread_id = nullptr);
    EnvironmentOutcome step(const Event& e, const std::size_t* thread_id = nullptr);
    EnvironmentOutcome sample(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    State get_next_state(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    NextStateDistribution get_next_state_distribution(const Memory& m, const Event& e, const std::size_t* thread_id = nullptr);
    Value get_transition_value(const Memory& m, const Event& e, const State& sp, const std::size_t* thread_id = nullptr);
    bool is_goal(const State& s, const std::size_t* thread_id = nullptr);
    bool is_terminal(const State& s, const std::size_t* thread_id = nullptr);

    template <typename Tfunction, typename ... Types>
    std::unique_ptr<py::object> do_launch(const std::size_t* thread_id,
                                          const Tfunction& func,
                                          const Types& ... args) {
        std::unique_ptr<py::object> id;
        nng::socket* conn = nullptr;
        {
            typename GilControl<Texecution>::Acquire acquire;
            try {
                if (thread_id) {
                    id = std::make_unique<py::object>(func(*_domain, args..., py::int_(*thread_id)));
                } else {
                    id = std::make_unique<py::object>(func(*_domain, args..., py::none()));
                }
                int did = py::cast<int>(*id);
                if (did >= 0) {
                    conn = _connections[(std::size_t) did].get();
                }
            } catch(const py::error_already_set* e) {
                Logger::error("SKDECIDE exception when asynchronously calling anonymous domain method: " + std::string(e->what()));
                std::runtime_error err(e->what());
                id.reset();
                delete e;
                throw err;
            }
        }
        if (conn) { // positive id returned (parallel execution, waiting for python process to return)
            try {
                nng::msg msg = conn->recv_msg();
                if (msg.body().size() != 1 || msg.body().data<char>()[0] != '0') { // error
                    typename GilControl<Texecution>::Acquire acquire;
                    id.reset();
                    std::string pyerr(msg.body().data<char>(), msg.body().size());
                    throw std::runtime_error("SKDECIDE exception: C++ parallel domain received an exception from Python parallel domain: " + pyerr);
                }
            } catch (const nng::exception& e) {
                std::string err_msg("SKDECIDE exception when waiting for a response from the python parallel domain: ");
                err_msg += e.who() + std::string(": ") + std::string(e.what());
                Logger::error(err_msg);
                typename GilControl<Texecution>::Acquire acquire;
                id.reset();
                throw std::runtime_error(err_msg);
            }
        } else {
            std::string err_msg("Unable to establish a connection with the Python parallel domain");
            Logger::error(err_msg);
            throw std::runtime_error(std::string("SKDECIDE exception: ") + err_msg);
        }
        typename GilControl<Texecution>::Acquire acquire;
        try {
            std::unique_ptr<py::object> r = std::make_unique<py::object>(_domain->attr("get_result")(*id));
            id.reset();
            return r;
        } catch(const py::error_already_set* e) {
            Logger::error("SKDECIDE exception when asynchronously calling the domain's get_result() method: " + std::string(e->what()));
            std::runtime_error err(e->what());
            id.reset();
            delete e;
            throw err;
        }
        id.reset();
        return std::make_unique<py::object>(py::none());
    }

    template <typename ... Types>
    std::unique_ptr<py::object> launch(const std::size_t* thread_id,
                                       const char* name,
                                       const Types& ... args) {
        return do_launch(thread_id, [&name](py::object& d, auto ... aargs){
            return d.attr(name)(aargs...);
        }, args...);
    }

    template <typename Tfunction, typename ... Types>
    std::unique_ptr<py::object> call(const std::size_t* thread_id,
                                     const Tfunction& func,
                                     const Types& ... args) {
        return do_launch(thread_id, func, args...);
    }

    struct NonTemplateMethods;
};

// === PythonDomainProxy implementation ===

template <typename Texecution, typename Tagent, typename Tobservability, typename Tcontrollability, typename Tmemory>
template <typename Tfunction, typename ... Types>
std::unique_ptr<py::object> PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability, Tmemory>::call(
        const std::size_t* thread_id, const Tfunction& func, const Types& ... args) {
    try {
        return _implementation->call(thread_id, func, args...);
    } catch(const std::exception& e) {
        Logger::error(std::string("SKDECIDE exception when calling anonymous domain method: ") + std::string(e.what()));
        throw;
    }
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_CALL_IMPL_HH
