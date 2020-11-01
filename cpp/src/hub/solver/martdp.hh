/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MARTDP_HH
#define SKDECIDE_MARTDP_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <stack>
#include <list>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include <boost/range/irange.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain,
          typename Texecution_policy = SequentialExecution>
class MARTDPSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::State::Agent SAgent;
    typedef typename Domain::State::Element AgentState;
    typedef typename Domain::Event Action;
    typedef typename Domain::Action::Agent AAgent;
    typedef typename Domain::Action::Element AgentAction;
    typedef typename Domain::Value Value;
    typedef typename Domain::EnvironmentOutcome EnvironmentOutcome;
    typedef Texecution_policy ExecutionPolicy;

    MARTDPSolver(Domain& domain,
                 const std::function<bool (Domain&, const State&)>& goal_checker,
                 const std::function<std::pair<Value, Action> (Domain&, const State&)>& heuristic,
                 std::size_t time_budget = 3600000,
                 std::size_t rollout_budget = 100000,
                 std::size_t max_depth = 1000,
                 std::size_t max_feasibility_trials = 0, // will then choose nb_agents
                 std::size_t nb_transition_smaples = 0, // will then choose 10*nb_agents
                 double discount = 1.0,
                 double action_choice_noise = 0.1,
                 double dead_end_cost = 10e4,
                 bool debug_logs = false)
        : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
          _time_budget(time_budget), _rollout_budget(rollout_budget), _max_depth(max_depth),
          _max_feasibility_trials(max_feasibility_trials), _nb_transition_samples(nb_transition_smaples),
          _discount(discount), _dead_end_cost(dead_end_cost),
          _debug_logs(debug_logs), _current_state(nullptr), _nb_rollouts(0),
          _nb_agents(0) {
              if (debug_logs) {
                spdlog::set_level(spdlog::level::debug);
              } else {
                  spdlog::set_level(spdlog::level::info);
              }

              std::random_device rd;
              _gen = std::make_unique<std::mt19937>(rd());
              _action_choice_noise_dist = std::bernoulli_distribution(action_choice_noise);
          }

    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear() {
        _graph.clear();
    }

    // solves from state s using heuristic function h
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print_type() + " MARTDP solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();

            if (_nb_agents != s.size()) {
                // We are solving a new problem.
                _graph.clear();
                _nb_agents = s.size();
                _agents_orders.resize(_nb_agents);

                for (std::size_t a = 0 ; a < _nb_agents ; a++) {
                    _agents_orders[a].resize(_nb_agents - 1);
                    for (std::size_t aa = 0 ; aa < _nb_agents - 1 ; aa++) {
                        if (aa < a) {
                            _agents_orders[a][aa] = aa;
                        } else {
                            _agents_orders[a][aa] = aa + 1;
                        }
                    }
                }

                _agents.clear();
                _agents.reserve(_nb_agents);

                for (auto a : s) {
                    _agents.push_back(a.agent());
                }

                if (_max_feasibility_trials == 0) {
                    _max_feasibility_trials = _nb_agents;
                }

                if (_nb_transition_samples == 0) {
                    _nb_transition_samples = 10 * _nb_agents;
                }
            }
            
            auto si = _graph.emplace(s);
            Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe

            if (si.second) {
                initialize_node(root_node);
            }

            if (_goal_checker(_domain, s)) { // problem already solved from this state (was present in _graph and already solved)
                if (_debug_logs) spdlog::debug("Found goal state " + s.print());
                return;
            }

            _nb_rollouts = 0;
                
            while ((elapsed_time(start_time) < _time_budget) &&
                   (_nb_rollouts < _rollout_budget)) {
                if (_debug_logs) spdlog::debug("Starting rollout " + StringConverter::from(_nb_rollouts) + ExecutionPolicy::print_thread());
                _nb_rollouts++;
                trial(&root_node, start_time);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            spdlog::info("MARTDP finished to solve from state " + s.print() +
                         " in " + StringConverter::from((double) duration / (double) 1e9) + " seconds with " +
                         StringConverter::from(_nb_rollouts) + " rollouts and visited " +
                         StringConverter::from(_graph.size()) + " states. ");
        } catch (const std::exception& e) {
            spdlog::error("MARTDP failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || !(si->best_action)) {
            return false;
        } else {
            return true;
        }
    }

    const Action& get_best_action(const State& s) {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || !(si->best_action)) {
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        } else {
            return *(si->best_action);
        }
    }

    double get_best_value(const State& s) const {
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        }
        return si->best_value;
    }

    std::size_t get_nb_of_explored_states() const {
        return _graph.size();
    }

    std::size_t get_nb_rollouts() const {
        return _nb_rollouts;
    }

    typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const {
        typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
        for (auto& n : _graph) {
            if (n.best_action) {
                p.insert(std::make_pair(n.state,
                                        std::make_pair(*n.best_action,
                                        (double) n.best_value)));
            }
        }
        return p;
    }

private :
    Domain& _domain;
    std::function<bool (Domain&, const State&)> _goal_checker;
    std::function<std::pair<Value, Action> (Domain&, const State&)> _heuristic;
    std::size_t _time_budget;
    std::size_t _rollout_budget;
    std::size_t _max_depth;
    std::size_t _max_feasibility_trials;
    std::size_t _nb_transition_samples;
    double _discount;
    double _dead_end_cost;
    bool _debug_logs;
    ExecutionPolicy _execution_policy;
    std::unique_ptr<std::mt19937> _gen;
    typename ExecutionPolicy::Mutex _gen_mutex;
    typename ExecutionPolicy::Mutex _time_mutex;

    struct Node {
        State state;
        std::unique_ptr<Action> best_action;
        double best_value;
        double goal;

        Node(const State& s)
        : state(s),
          best_value(std::numeric_limits<double>::infinity()),
          goal(false) {}
        
        struct Key {
            const State& operator()(const Node& sn) const { return sn.state; }
        };
    };

    typedef typename SetTypeDeducer<Node, State>::Set Graph;
    Graph _graph;
    Node* _current_state;
    std::size_t _nb_rollouts;
    std::size_t _nb_agents;
    std::vector<SAgent> _agents;
    std::vector<std::vector<std::size_t>> _agents_orders;
    std::bernoulli_distribution _action_choice_noise_dist;

    std::size_t elapsed_time(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
        std::size_t milliseconds_duration;
        _execution_policy.protect([&milliseconds_duration, &start_time](){
            milliseconds_duration = static_cast<std::size_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time
                ).count()
            );
        }, _time_mutex);
        return milliseconds_duration;
    }

    std::unique_ptr<Action> greedy_action(Node* s) {
        if (_debug_logs) spdlog::debug("Updating state " + s->state.print() +
                                       ExecutionPolicy::print_thread());
        
        double best_value = std::numeric_limits<double>::infinity();
        std::unique_ptr<Action> best_action;
        
        for (std::size_t agent = 0 ; agent < _nb_agents ; agent++) {
            auto agent_applicable_actions = _domain.get_agent_applicable_actions(s->state, Action(), _agents[agent]).get_elements();

            if (agent_applicable_actions.empty()) {
                continue;
            } else {
                for (auto action : agent_applicable_actions) {
                    std::unique_ptr<Action> agent_actions = std::make_unique<Action>();
                    (*agent_actions)[_agents[agent]] = action;

                    // try to find a feasible assignment
                    bool feasible = false;
                    std::size_t feasibility_trial = 0;
                    
                    while (!feasible && feasibility_trial < _max_feasibility_trials) {
                        std::shuffle(_agents_orders[agent].begin(), _agents_orders[agent].end(), *_gen);
                        feasibility_trial++;
                        feasible = true;

                        // construct the joint action
                        for (auto other_agent : _agents_orders[agent]) {
                            auto other_agent_aa = _domain.get_agent_applicable_actions(s->state, *agent_actions, _agents[other_agent]);

                            if (other_agent_aa.empty()) {
                                feasible = false;
                                break;
                            }
                            
                            // Is the agent's optimal action applicable ?
                            if (s->best_action) {
                                if (other_agent_aa.contains((*(s->best_action))[_agents[other_agent]])) {
                                    // choose it with high probability
                                    if (!(_action_choice_noise_dist(*_gen))) {
                                        (*agent_actions)[_agents[other_agent]] = (*s->best_action)[_agents[other_agent]];
                                    } else {
                                        (*agent_actions)[_agents[other_agent]] = other_agent_aa.sample();
                                    }
                                }
                            } else {
                                (*agent_actions)[_agents[other_agent]] = other_agent_aa.sample();
                            }
                        }
                    }

                    if (feasible) {
                        // make the transition and compute Q-value
                        double qval = q_value(s->state, *agent_actions);

                        if (qval < best_value) {
                            best_value = qval;
                            best_action = std::move(agent_actions);
                        }
                    }
                }
            }
        }

        if (_debug_logs) {
            if (best_action) {
                spdlog::debug("Greedy action of state " + s->state.print() + ": " +
                              best_action->print() +
                              " with value " + StringConverter::from(best_value));
            } else {
                spdlog::debug("State " + s->state.print() +
                              " is a dead-end (no feasible actions found)");
            }
        }

        s->best_action = std::move(best_action);
        s->best_value = best_value;
        return best_action; // best_action is nullptr if state is a dead-end
    }

    double q_value(const State& state, const Action& action) {
        double q_value = 0;
        for (std::size_t transition_sample = 0 ; transition_sample < _nb_transition_samples ; transition_sample++) {
            EnvironmentOutcome outcome = _domain.sample(state, action);
            Value tval = outcome.transition_value();
            auto i = _graph.emplace(outcome.observation());
            Node& next_node = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
            if (i.second) { // new node
                initialize_node(next_node);
            }
            double sample_value = 0.0;
            for (auto a : _agents) {
                sample_value += tval[a].cost();
            }
            sample_value += _discount * next_node.best_value;
            q_value += sample_value;
        }
        q_value /= _nb_transition_samples;

        if (_debug_logs) spdlog::debug("Updated Q-value of action " + action.print() +
                                        " with value " + StringConverter::from(q_value));

        return q_value;
    }

    Node* pick_next_state(const Node* s, const Action& a) {
        EnvironmentOutcome outcome = _domain.sample(s->state, a);
        auto i = _graph.emplace(outcome.observation());
        Node& next_node = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
        if (i.second) { // new node
            initialize_node(next_node);
        }
        if (_debug_logs) spdlog::debug("Picked next state " + next_node.state.print() +
                                       " from state " + s->state.print() +
                                        " and action " + a.print());
        return &next_node;
    }

    void initialize_node(Node& n) {
        if (_debug_logs) {
            spdlog::debug("Initiliazing new state node " + n.state.print());
        }
        n.best_value = 0.0;
        n.goal = _goal_checker(_domain, n.state);
        if (!n.goal) {
            auto h = _heuristic(_domain, n.state);
            n.best_action = std::make_unique<Action>();
            for (auto a : _agents) {
                n.best_value += h.first[a].cost();
                (*n.best_action)[a] = h.second[a];
            }
        }
    }

    void trial(Node* s,
               const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
        Node* cs = s;
        std::size_t depth = 0;
        bool found_goal = false;

        while (!found_goal &&
               (elapsed_time(start_time) < _time_budget) &&
               (depth < _max_depth)) {
            depth++;
            if (cs->goal) {
                if (_debug_logs) spdlog::debug("Found goal state " + cs->state.print() +
                                                ExecutionPolicy::print_thread());
                found_goal = true;
            }

            auto action = greedy_action(cs);

            if (!action) { // dead-end state
                cs->best_value = _dead_end_cost;
            } else {
                cs = pick_next_state(cs, *action);
            }
        }
    }
};

} // namespace skdecide

#endif // SKDECIDE_MA_RTDP_HH
