/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the skdecide implementation of LRTDP from the paper
 * "Labeled RTDP: Improving the Convergence of Real-Time Dynamic
 * Programming" from Bonet and Geffner (ICAPS 2003)
 */
#ifndef SKDECIDE_LRTDP_HH
#define SKDECIDE_LRTDP_HH

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
class LRTDPSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Texecution_policy ExecutionPolicy;

    LRTDPSolver(Domain& domain,
                const std::function<bool (Domain&, const State&, const std::size_t*)>& goal_checker,
                const std::function<double (Domain&, const State&, const std::size_t*)>& heuristic,
                bool use_labels = true,
                std::size_t time_budget = 3600000,
                std::size_t rollout_budget = 100000,
                std::size_t max_depth = 1000,
                double discount = 1.0,
                double epsilon = 0.001,
                bool online_node_garbage = false,
                bool debug_logs = false)
        : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic), _use_labels(use_labels),
          _time_budget(time_budget), _rollout_budget(rollout_budget), _max_depth(max_depth),
          _discount(discount), _epsilon(epsilon), _online_node_garbage(online_node_garbage),
          _debug_logs(debug_logs), _current_state(nullptr), _nb_rollouts(0) {
              if (debug_logs) {
                spdlog::set_level(spdlog::level::debug);
              } else {
                  spdlog::set_level(spdlog::level::info);
              }

              std::random_device rd;
              _gen = std::make_unique<std::mt19937>(rd());
          }

    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear() {
        _graph.clear();
    }

    // solves from state s using heuristic function h
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print_type() + " LRTDP solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto si = _graph.emplace(s);
            StateNode& root_node = const_cast<StateNode&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe

            if (si.second) {
                root_node.best_value = _heuristic(_domain, s, nullptr);
            }

            if (root_node.solved || _goal_checker(_domain, s, nullptr)) { // problem already solved from this state (was present in _graph and already solved)
                if (_debug_logs) spdlog::debug("Found goal state " + s.print());
                return;
            }

            _nb_rollouts = 0;
            boost::integer_range<std::size_t> parallel_rollouts(0, _domain.get_parallel_capacity());

            std::for_each(ExecutionPolicy::policy, parallel_rollouts.begin(), parallel_rollouts.end(),
                            [this, &start_time, &root_node] (const std::size_t& thread_id) {
                
                while ((!_use_labels || !root_node.solved) &&
                       (elapsed_time(start_time) < _time_budget) &&
                       (_nb_rollouts < _rollout_budget)) {
                    if (_debug_logs) spdlog::debug("Starting rollout " + StringConverter::from(_nb_rollouts) + ExecutionPolicy::print_thread());
                    _nb_rollouts++;
                    trial(&root_node, start_time, &thread_id);
                }
            });

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            spdlog::info("LRTDP finished to solve from state " + s.print() +
                         " in " + StringConverter::from((double) duration / (double) 1e9) + " seconds with " +
                         StringConverter::from(_nb_rollouts) + " rollouts and visited " +
                         StringConverter::from(_graph.size()) + " states. ");
        } catch (const std::exception& e) {
            spdlog::error("LRTDP failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            // /!\ does not mean the state is solved!
            return false;
        } else {
            return true;
        }
    }

    const Action& get_best_action(const State& s) {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        } else {
            if (_debug_logs) {
                    std::string str = "(";
                    for (const auto& o : si->best_action->outcomes) {
                        str += "\n    " + std::get<2>(o)->state.print();
                    }
                    str += "\n)";
                    spdlog::debug("Best action's outcomes:\n" + str);
                }
            if (_online_node_garbage && _current_state) {
                std::unordered_set<StateNode*> root_subgraph, child_subgraph;
                compute_reachable_subgraph(_current_state, root_subgraph);
                compute_reachable_subgraph(const_cast<StateNode*>(&(*si)), child_subgraph); // we won't change the real key (StateNode::state) so we are safe
                remove_subgraph(root_subgraph, child_subgraph);
            }
            _current_state = const_cast<StateNode*>(&(*si)); // we won't change the real key (StateNode::state) so we are safe
            return si->best_action->action;
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
            if (n.best_action != nullptr) {
                p.insert(std::make_pair(n.state,
                                        std::make_pair(n.best_action->action,
                                        (double) n.best_value)));
            }
        }
        return p;
    }

private :
    typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
    typedef typename ExecutionPolicy::template atomic<double> atomic_double;
    typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;
    
    Domain& _domain;
    std::function<bool (Domain&, const State&, const std::size_t*)> _goal_checker;
    std::function<double (Domain&, const State&, const std::size_t*)> _heuristic;
    bool _use_labels;
    atomic_size_t _time_budget;
    atomic_size_t _rollout_budget;
    atomic_size_t _max_depth;
    atomic_double _discount;
    atomic_double _epsilon;
    bool _online_node_garbage;
    atomic_bool _debug_logs;
    ExecutionPolicy _execution_policy;
    std::unique_ptr<std::mt19937> _gen;
    typename ExecutionPolicy::Mutex _gen_mutex;
    typename ExecutionPolicy::Mutex _time_mutex;

    struct ActionNode;

    struct StateNode {
        State state;
        std::list<std::unique_ptr<ActionNode>> actions;
        ActionNode* best_action;
        atomic_double best_value;
        atomic_double goal;
        atomic_bool solved;
        typename ExecutionPolicy::Mutex mutex;

        StateNode(const State& s)
            : state(s), best_action(nullptr),
              best_value(std::numeric_limits<double>::infinity()),
              goal(false), solved(false) {}
        
        struct Key {
            const State& operator()(const StateNode& sn) const { return sn.state; }
        };
    };

    struct ActionNode {
        Action action;
        std::vector<std::tuple<double, double, StateNode*>> outcomes; // next state nodes owned by _graph
        std::discrete_distribution<> dist;
        atomic_double value;

        ActionNode(const Action& a)
            : action(a), value(std::numeric_limits<double>::infinity()) {}
    };

    typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
    Graph _graph;
    StateNode* _current_state;
    atomic_size_t _nb_rollouts;

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

    void expand(StateNode* s, const std::size_t* thread_id) {
        if (_debug_logs) spdlog::debug("Expanding state " + s->state.print() + ExecutionPolicy::print_thread());
        auto applicable_actions = _domain.get_applicable_actions(s->state, thread_id).get_elements();

        for (auto a : applicable_actions) {
            if (_debug_logs) spdlog::debug("Current expanded action: " + a.print() + ExecutionPolicy::print_thread());
            s->actions.push_back(std::make_unique<ActionNode>(a));
            ActionNode& an = *(s->actions.back());
            auto next_states = _domain.get_next_state_distribution(s->state, a, thread_id).get_values();
            std::vector<double> outcome_weights;

            for (auto ns : next_states) {
                std::pair<typename Graph::iterator, bool> i;
                _execution_policy.protect([this, &i, &ns]{
                    i = _graph.emplace(ns.state());
                });
                StateNode& next_node = const_cast<StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                an.outcomes.push_back(std::make_tuple(ns.probability(), _domain.get_transition_cost(s->state, a, next_node.state, thread_id), &next_node));
                outcome_weights.push_back(std::get<0>(an.outcomes.back()));
                if (_debug_logs) spdlog::debug("Current next state expansion: " + next_node.state.print() + ExecutionPolicy::print_thread());

                if (i.second) { // new node
                    if (_goal_checker(_domain, next_node.state, thread_id)) {
                        if (_debug_logs) spdlog::debug("Found goal state " + next_node.state.print() + ExecutionPolicy::print_thread());
                        next_node.goal = true;
                        next_node.solved = true;
                        next_node.best_value = 0.0;
                    } else {
                        next_node.best_value = _heuristic(_domain, next_node.state, thread_id);
                        if (_debug_logs) spdlog::debug("New state " + next_node.state.print() +
                                                       " with heuristic value " + StringConverter::from(next_node.best_value) +
                                                       ExecutionPolicy::print_thread());
                    }
                }
            }

            an.dist = std::discrete_distribution<>(outcome_weights.begin(), outcome_weights.end());
        }
    }

    double q_value(ActionNode* a) {
        a->value = 0;
        for (const auto& o : a->outcomes) {
            a->value = a->value + (std::get<0>(o) * (std::get<1>(o) + (_discount * std::get<2>(o)->best_value)));
        }
        if (_debug_logs) spdlog::debug("Updated Q-value of action " + a->action.print() +
                                        " with value " + StringConverter::from(a->value) +
                                        ExecutionPolicy::print_thread());
        return a->value;
    }

    ActionNode* greedy_action(StateNode* s, const std::size_t* thread_id) {
        double best_value = std::numeric_limits<double>::infinity();
        ActionNode* best_action = nullptr;

        if (s->actions.empty()) {
            expand(s, thread_id);
        }

        for (auto& a : s->actions) {
            if (q_value(a.get()) < best_value) {
                best_value = a->value;
                best_action = a.get();
            }
        }

        if (_debug_logs) {
            spdlog::debug("Greedy action of state " + s->state.print() + ": " +
                            best_action->action.print() + " with value " + StringConverter::from(best_value) +
                            ExecutionPolicy::print_thread());
        }

        return best_action;
    }

    void update(StateNode* s, const std::size_t* thread_id) {
        if (_debug_logs) spdlog::debug("Updating state " + s->state.print() +
                                       ExecutionPolicy::print_thread());
        s->best_action = greedy_action(s, thread_id);
        s->best_value = (double) s->best_action->value;
    }

    StateNode* pick_next_state(ActionNode* a) {
        StateNode* s = nullptr;
        _execution_policy.protect([&a, &s, this](){
                s = std::get<2>(a->outcomes[a->dist(*_gen)]);
                if (_debug_logs) spdlog::debug("Picked next state " + s->state.print() +
                                               " from action " + a->action.print() +
                                               ExecutionPolicy::print_thread());
        }, _gen_mutex);
        return s;
    }

    double residual(StateNode* s, const std::size_t* thread_id) {
        s->best_action = greedy_action(s, thread_id);
        double res = std::fabs(s->best_value - s->best_action->value);
        if (_debug_logs) spdlog::debug("State " + s->state.print() +
                                       " has residual " + StringConverter::from(res) +
                                       ExecutionPolicy::print_thread());
        return res;
    }

    bool check_solved(StateNode* s,
                      const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time,
                      const std::size_t* thread_id) {
        if (_debug_logs) {
            _execution_policy.protect([&s](){
                spdlog::debug("Checking solved status of State " + s->state.print() +
                              ExecutionPolicy::print_thread());
            }, s->mutex);
        }

        bool rv = true;
        std::stack<StateNode*> open;
        std::stack<StateNode*> closed;
        std::unordered_set<StateNode*> visited;
        std::size_t depth = 0;

        if (!(s->solved)) {
            open.push(s);
            visited.insert(s);
        }

        while (!open.empty() &&
               (elapsed_time(start_time) < _time_budget) &&
               (depth < _max_depth)) {
            depth++;
            StateNode* cs = open.top();
            open.pop();
            closed.push(cs);
            visited.insert(cs);

            _execution_policy.protect([this, &cs, &rv, &open, &visited, &thread_id](){
                if (residual(cs, thread_id) > _epsilon) {
                    rv = false;
                    return;
                }

                ActionNode* a = cs->best_action; // best action updated when calling residual(cs, thread_id)
                for (const auto& o : a->outcomes) {
                    StateNode* ns = std::get<2>(o);
                    if (!(ns->solved) && (visited.find(ns) == visited.end())) {
                        open.push(ns);
                    }
                }
            }, cs->mutex);
        }

        auto e_time = elapsed_time(start_time);
        rv = rv && ((e_time < _time_budget) ||
                    ((e_time >= _time_budget) && open.empty()));

        if (rv) {
            while (!closed.empty()) {
                closed.top()->solved = true;
                closed.pop();
            }
        } else {
            while (!closed.empty() && (elapsed_time(start_time) < _time_budget)) {
                _execution_policy.protect([this, &closed, &thread_id](){
                    update(closed.top(), thread_id);
                }, closed.top()->mutex);
                closed.pop();
            }
        }

        if (_debug_logs) {
            _execution_policy.protect([&s, &rv](){
                spdlog::debug("State " + s->state.print() + " is " + (rv?(""):("not")) + " solved." +
                              ExecutionPolicy::print_thread());
            }, s->mutex);
        }

        return rv;
    }

    void trial(StateNode* s,
               const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time,
               const std::size_t* thread_id) {
        std::stack<StateNode*> visited;
        StateNode* cs = s;
        std::size_t depth = 0;
        bool found_goal = false;

        while ((!_use_labels || !(cs->solved)) &&
               !found_goal &&
               (elapsed_time(start_time) < _time_budget) &&
               (depth < _max_depth)) {
            depth++;
            visited.push(cs);
            _execution_policy.protect([this, &cs, &found_goal, &thread_id](){
                if (cs->goal) {
                    if (_debug_logs) spdlog::debug("Found goal state " + cs->state.print() +
                                                   ExecutionPolicy::print_thread());
                    found_goal = true;
                }

                update(cs, thread_id);
                cs = pick_next_state(cs->best_action);
            }, cs->mutex);
        }

        while (_use_labels && !visited.empty() && (elapsed_time(start_time) < _time_budget)) {
            cs = visited.top();
            visited.pop();

            if (!check_solved(cs, start_time, thread_id)) {
                break;
            }
        }
    }

    void compute_reachable_subgraph(StateNode* node, std::unordered_set<StateNode*>& subgraph) {
        std::unordered_set<StateNode*> frontier;
        frontier.insert(node);
        subgraph.insert(node);
        while(!frontier.empty()) {
            std::unordered_set<StateNode*> new_frontier;
            for (auto& n : frontier) {
                for (auto& action : n->actions) {
                    for (auto& outcome : action->outcomes) {
                        if (subgraph.find(std::get<2>(outcome)) == subgraph.end()) {
                            new_frontier.insert(std::get<2>(outcome));
                            subgraph.insert(std::get<2>(outcome));
                        }
                    }
                }
            }
            frontier = new_frontier;
        }
    }

    void remove_subgraph(std::unordered_set<StateNode*>& root_subgraph, std::unordered_set<StateNode*>& child_subgraph) {
        for (auto& n : root_subgraph) {
            if (child_subgraph.find(n) == child_subgraph.end()) {
                _graph.erase(StateNode(n->state));
            }
        }
    }
};

} // namespace skdecide

#endif // SKDECIDE_LRTDP_HH
