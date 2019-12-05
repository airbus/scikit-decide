/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_MCTS_HH
#define AIRLAPS_MCTS_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace airlaps {


/** Default tree policy as used in UCT */
class DefaultTreePolicy {
public :
    DefaultTreePolicy() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    template <typename Tsolver, typename Texpander, typename TactionSelector>
    typename Tsolver::StateNode* operator()(Tsolver& solver,
                                            const Texpander& expander,
                                            const TactionSelector& action_selector,
                                            typename Tsolver::StateNode& n,
                                            std::size_t& d) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Launching default tree policy from state " + n.state.print()); }
            typename Tsolver::StateNode* current_node = &n;
            while(!(current_node->terminal) && d < solver.max_depth()) {
                typename Tsolver::StateNode* next_node = expander(solver, *current_node);
                d++;
                if (next_node == nullptr) { // node fully expanded
                    typename Tsolver::ActionNode* action = action_selector(solver, *current_node);
                    if (action == nullptr) {
                        throw std::runtime_error("AIRLAPS exception: no best action found in state " + current_node->state.print());
                    } else {
                        current_node = action->dist_to_outcome[action->dist(gen)]->first;
                    }
                } else {
                    current_node = next_node;
                    break;
                }
            }
            return current_node;
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when simulating the tree policy from state " + n.state.print() + ": " + e.what());
            throw;
        }
    }

private :
    mutable std::mt19937 gen;
};


/** Test if a given node needs to be expanded by assuming that applicable actions and next states
 *  can be enumerated. Returns nullptr if all actions and outcomes have already tried, otherwise a
 *  sampled unvisited outcome according to its probability (among only unvisited outcomes).
 *  REQUIREMENTS: returns nullptr if all actions have been already tried, and set the terminal
 *  flag of the returned next state
 */
class EnumerationExpand {
public :
    EnumerationExpand() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    template <typename Tsolver>
    typename Tsolver::StateNode* operator()(Tsolver& solver, typename Tsolver::StateNode& n) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Test expansion of state " + n.state.print()); }
            if (n.expanded) {
                if (solver.debug_logs()) { spdlog::debug("State already fully expanded"); }
                return nullptr;
            }
            // Generate applicable actions if not already done
            if (n.actions.empty()) {
                if (solver.debug_logs()) { spdlog::debug("State never expanded, generating all next actions"); }
                auto applicable_actions = solver.domain().get_applicable_actions(n.state)->get_elements();
                for (const auto& a : applicable_actions) {
                    n.actions.emplace_back(std::make_unique<typename Tsolver::ActionNode>(a));
                    n.actions.back()->parent = &n;
                }
            }
            // Check for untried outcomes
            if (solver.debug_logs()) { spdlog::debug("Checking for untried outcomes..."); }
            std::vector<std::pair<typename Tsolver::ActionNode*, typename Tsolver::StateNode*>> untried_outcomes;
            std::vector<double> weights;
            for (auto& a : n.actions) {
                if (a->outcomes.empty()) {
                    untried_outcomes.push_back(std::make_pair(a.get(), nullptr));
                    weights.push_back(1.0);
                } else {
                    // Check if there are next states that have been never visited
                    std::vector<double> probs = a->dist.probabilities();
                    for (std::size_t p = 0 ; p < probs.size() ; p++) {
                        typename Tsolver::StateNode* on = a->dist_to_outcome[p]->first;
                        if (on->visits_count == 0) {
                            untried_outcomes.push_back(std::make_pair(a.get(), on));
                            weights.push_back(probs[p]);
                        }
                    }
                }
            }
            if (untried_outcomes.empty()) { // nothing to expand
                if (solver.debug_logs()) { spdlog::debug("All outcomes already tried"); }
                n.expanded = true;
                return nullptr;
            } else {
                std::discrete_distribution<> odist(weights.begin(), weights.end());
                auto& uo = untried_outcomes[odist(gen)];
                if (uo.second == nullptr) { // unexpanded action
                    if (solver.debug_logs()) { spdlog::debug("Found one unexpanded action: " + uo.first->action.print()); }
                    // Generate the next states of this action
                    auto next_states = solver.domain().get_next_state_distribution(n.state, uo.first->action)->get_values();
                    typename Tsolver::ActionNode& action = *uo.first;
                    untried_outcomes.clear();
                    weights.clear();
                    std::vector<double> outcome_weights;
                    for (const auto& ns : next_states) {
                        typename Tsolver::Domain::OutcomeExtractor oe(ns);
                        auto i = solver.graph().emplace(oe.state());
                        typename Tsolver::StateNode& next_node = const_cast<typename Tsolver::StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                        auto ii = action.outcomes.insert(std::make_pair(&next_node,
                                                                        solver.domain().get_transition_reward(n.state, action.action, next_node.state)));
                        action.dist_to_outcome.insert(std::make_pair(outcome_weights.size(), ii));
                        outcome_weights.push_back(oe.probability());
                        next_node.parents.push_back(&action);
                        if (i.second) { // new node
                            next_node.terminal = solver.domain().is_terminal(next_node.state);
                        }
                        if (next_node.actions.empty()) {
                            if (solver.debug_logs()) spdlog::debug("Candidate next state: " + next_node.state.print());
                            untried_outcomes.push_back(std::make_pair(&action, &next_node));
                            weights.push_back(oe.probability());
                        }
                    }
                    // Record the action's outcomes distribution
                    action.dist = std::discrete_distribution<>(outcome_weights.begin(), outcome_weights.end());
                    // Pick a random next state
                    if (untried_outcomes.empty()) {
                        // All next states already visited => pick a random next state using action.dist
                        return action.dist_to_outcome[action.dist(gen)]->first;
                    } else {
                        // Pick a random next state among untried ones
                        odist = std::discrete_distribution<>(weights.begin(), weights.end());
                        return untried_outcomes[odist(gen)].second;
                    }
                } else { // expanded action, just return the selected next state
                    if (solver.debug_logs()) { spdlog::debug("Found one untried outcome: action " + uo.first->action.print() +
                                                             " and next state " + uo.second->state.print()); }
                    return uo.second;
                }
            }
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when expanding state " + n.state.print() + ": " + e.what());
            throw;
        }
    }

private :
    mutable std::mt19937 gen;
};


/** UCB1 Best Child */
class UCB1BestChild {
public :
    // 1/sqrt(2) is a good compromise for rewards in [0;1]
    UCB1BestChild(double ucb_constant = 1.0 / std::sqrt(2.0))
    : _ucb_constant(ucb_constant) {}

    template <typename Tsolver>
    typename Tsolver::ActionNode* operator()(Tsolver& solver, const typename Tsolver::StateNode& n) const {
        double best_value = -std::numeric_limits<double>::max();
        typename Tsolver::ActionNode* best_action = nullptr;
        for (const auto& a : n.actions) {
            double tentative_value = a->value + (_ucb_constant * std::sqrt((2.0 * std::log(n.visits_count)) / a->visits_count));
            if (tentative_value > best_value) {
                best_value = tentative_value;
                best_action = &(*a);
            }
        }
        if (solver.debug_logs()) { spdlog::debug("UCB1 selection from state " + n.state.print() +
                                                 ": value=" + std::to_string(best_value) +
                                                 ", action=" + best_action->action.print()); }
        return best_action;
    }

private :
    double _ucb_constant;
};


/** Random default policy */
class RandomDefaultPolicy {
public :
    template <typename Tsolver>
    void operator()(Tsolver& solver, typename Tsolver::StateNode& n, std::size_t d) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Launching random default policy from state " + n.state.print()); }
            typename Tsolver::Domain::State current_state = n.state;
            std::size_t current_depth = d;
            double reward = 0.0;
            double gamma_n = 1.0;
            while(!solver.domain().is_terminal(current_state) && current_depth < solver.max_depth()) {
                std::unique_ptr<typename Tsolver::Domain::Action> action = solver.domain().get_applicable_actions(current_state)->sample();
                typename std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> o = solver.domain().sample(
                    current_state,
                    *action
                );
                reward += gamma_n * (o->reward());
                gamma_n *= solver.discount();
                current_state = o->state();
                current_depth++;
                if (solver.debug_logs()) { spdlog::debug("Sampled transition: action=" + action->print() +
                                                        ", next state=" + current_state.print() +
                                                        ", reward=" + std::to_string(o->reward())); }
            }
            // since we can come to state n after exhausting the depth, n might be already visited
            // so don't erase its value but rather update it
            n.value = ((n.visits_count * n.value)  + reward) / ((double) (n.visits_count + 1));
            n.visits_count += 1;
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when simulating the random default policy from state " + n.state.print() + ": " + e.what());
            throw;
        }
    }
};


/** Enumeration backup: enumerate all next states to update Q values */
struct EnumerationBackup {
    template <typename Tsolver>
    void operator()(Tsolver& solver, typename Tsolver::StateNode& n) const {
        if (solver.debug_logs()) { spdlog::debug("Back-propagating values from state " + n.state.print()); }
        std::size_t depth = 0; // used to prevent infinite loop in case of cycles
        std::unordered_set<typename Tsolver::StateNode*> frontier;
        frontier.insert(&n);
        while (!frontier.empty() && depth <= solver.max_depth()) {
            depth++;
            std::unordered_set<typename Tsolver::StateNode*> new_frontier;
            for (auto& f : frontier) {
                for (auto& a : f->parents) {
                    double q_value = 0.0;
                    auto range = a->outcomes.equal_range(f); // get all same next states (in case of redundancies)
                    for (auto i = range.first ; i != range.second ; ++i) {
                        q_value += i->second + (solver.discount() * i->first->value);
                    }
                    a->value = (((a->visits_count) * (a->value))  + q_value) / ((double) (a->visits_count + 1));
                    a->visits_count += 1;
                    typename Tsolver::StateNode* parent_node = a->parent;
                    parent_node->value = (((parent_node->visits_count) * (parent_node->value))  + (a->value)) / ((double) (parent_node->visits_count + 1));
                    parent_node->visits_count += 1;
                    new_frontier.insert(parent_node);
                    if (solver.debug_logs()) { spdlog::debug("Updating state " + parent_node->state.print() +
                                                             ": value=" + std::to_string(parent_node->value) +
                                                             ", visits=" + std::to_string(parent_node->visits_count)); }
                }
            }
            frontier = new_frontier;
        }
    }
};


template <typename Tdomain,
          typename Texecution_policy = ParallelExecution,
          typename TtreePolicy = DefaultTreePolicy,
          typename Texpander = EnumerationExpand,
          typename TactionSelector = UCB1BestChild,
          typename TdefaultPolicy = RandomDefaultPolicy,
          typename TbackPropagator = EnumerationBackup>
class MCTSSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef TtreePolicy TreePolicy;
    typedef Texpander Expander;
    typedef TactionSelector ActionSelector;
    typedef TdefaultPolicy DefaultPolicy;
    typedef TbackPropagator BackPropagator;
    typedef Texecution_policy ExecutionPolicy;

    struct ActionNode;

    struct StateNode {
        State state;
        bool terminal;
        bool expanded;
        std::list<std::unique_ptr<ActionNode>> actions;
        double value;
        std::size_t visits_count;
        std::list<ActionNode*> parents;

        StateNode(const State& s)
            : state(s), terminal(false), expanded(false),
              value(0.0), visits_count(0) {}
        
        struct Key {
            const State& operator()(const StateNode& sn) const { return sn.state; }
        };
    };

    struct ActionNode {
        Action action;
        std::unordered_multimap<StateNode*, double> outcomes; // next state nodes owned by _graph
        std::unordered_map<std::size_t, typename std::unordered_multimap<StateNode*, double>::iterator> dist_to_outcome;
        std::discrete_distribution<> dist;
        double value;
        std::size_t visits_count;
        StateNode* parent;

        ActionNode(const Action& a)
            : action(a), value(0.0), visits_count(0), parent(nullptr) {}
    };

    typedef typename SetTypeDeducer<StateNode, State>::Set Graph;

    MCTSSolver(Domain& domain,
               std::size_t time_budget = 3600000,
               std::size_t rollout_budget = 100000,
               std::size_t max_depth = 1000,
               double discount = 1.0,
               bool debug_logs = false,
               const TreePolicy& tree_policy = TreePolicy(),
               const Expander& expander = Expander(),
               const ActionSelector& action_selector = ActionSelector(),
               const DefaultPolicy& default_policy = DefaultPolicy(),
               const BackPropagator& back_propagator = BackPropagator())
    : _domain(domain),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _discount(discount), _nb_rollouts(0),
      _debug_logs(debug_logs), _tree_policy(tree_policy), _expander(expander),
      _action_selector(action_selector), _default_policy(default_policy),
      _back_propagator(back_propagator) {
        if (debug_logs) {
            spdlog::set_level(spdlog::level::debug);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
    }

    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear() {
        _graph.clear();
    }

    // solves from state s
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print() + " MCTS solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();
            _nb_rollouts = 0;

            // Get the root node
            auto si = _graph.emplace(s);
            StateNode& root_node = const_cast<StateNode&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe

            while (static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) < _time_budget &&
                       _nb_rollouts < _rollout_budget) {
                
                std::size_t depth = 0;
                StateNode* sn = _tree_policy(*this, _expander, _action_selector, root_node, depth);
                _default_policy(*this, *sn, depth);
                _back_propagator(*this, *sn);
                _nb_rollouts++;

            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            spdlog::info("MCTS finished to solve from state " + s.print() +
                         " in " + std::to_string((double) duration / (double) 1e9) + " seconds with " +
                         std::to_string(_nb_rollouts) + " rollouts.");
        } catch (const std::exception& e) {
            spdlog::error("MCTS failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) const {
        auto si = _graph.find(StateNode(s));
        if (si == _graph.end()) {
            return false;
        } else {
            return _action_selector(*this, *si) != nullptr;
        }
    }

    Action get_best_action(const State& s) {
        auto si = _graph.find(StateNode(s));
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = _action_selector(*this, *si);
        }
        if (action == nullptr) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        } else {
            return action->action;
        }
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(StateNode(s));
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = _action_selector(*this, *si);
        }
        if (action == nullptr) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        } else {
            return action->value;
        }
    }

    std::size_t nb_of_explored_states() const {
        return _graph.size();
    }

    std::size_t nb_rollouts() const {
        return _nb_rollouts;
    }

    Domain& domain() { return _domain; }

    std::size_t time_budget() const { return _time_budget; }

    std::size_t rollout_budget() const { return _rollout_budget; }

    std::size_t max_depth() const { return _max_depth; }

    double discount() const { return _discount; }

    const TreePolicy& tree_policy() { return _tree_policy; }

    const Expander& expander() { return _expander; }

    const ActionSelector& action_selector() { return _action_selector; }

    const DefaultPolicy& default_policy() { return _default_policy; }

    const BackPropagator& back_propagator() { return _back_propagator; }

    Graph& graph() { return _graph; }

    bool debug_logs() const { return _debug_logs; }

private :

    Domain& _domain;
    std::size_t _time_budget;
    std::size_t _rollout_budget;
    std::size_t _max_depth;
    double _discount;
    std::size_t _nb_rollouts;
    bool _debug_logs;
    TreePolicy _tree_policy;
    Expander _expander;
    ActionSelector _action_selector;
    DefaultPolicy _default_policy;
    BackPropagator _back_propagator;

    Graph _graph;
}; // MCTSSolver class

/** UCT is MCTS with the default template options */
template <typename Tdomain, typename Texecution_policy, typename ...T>
using UCTSolver = MCTSSolver<Tdomain, Texecution_policy, T...>;

} // namespace airlaps

#endif // AIRLAPS_MCTS_HH
