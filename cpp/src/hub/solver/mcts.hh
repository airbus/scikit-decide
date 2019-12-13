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

/** Use Environment domain knowledge for transitions */
struct StepTransitionMode {
    template <typename Tsolver>
    void init_rollout(Tsolver& solver) const {
        solver.domain().reset();
        std::for_each(solver.action_prefix().begin(), solver.action_prefix().end(),
                      [&solver](const typename Tsolver::Domain::Event& a){solver.domain().step(a);});
    }

    template <typename Tsolver>
    std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> random_next_outcome(
            Tsolver& solver,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().step(action);
    }

    template <typename Tsolver>
    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            typename Tsolver::ActionNode& action) const {
        auto outcome = solver.domain().step(action.action);
        auto si = solver.graph().find(typename Tsolver::StateNode(outcome->state()));
        if (si == solver.graph().end()) {
            return nullptr;
        } else {
            // we won't change the real key (ActionNode::action) so we are safe
            return &const_cast<typename Tsolver::StateNode&>(*si);
        }
    }
};


/** Use Simulation domain knowledge for transitions */
struct SampleTransitionMode {
    template <typename Tsolver>
    void init_rollout(Tsolver& solver) const {}

    template <typename Tsolver>
    std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> random_next_outcome(
            Tsolver& solver,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().sample(state, action);
    }

    template <typename Tsolver>
    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            typename Tsolver::ActionNode& action) const {
        return action.dist_to_outcome[action.dist(solver.gen())]->first;
    }
};


/** Use uncertain transitions domain knowledge for transitions */
struct DistributionTransitionMode {
    template <typename Tsolver>
    void init_rollout(Tsolver& solver) const {}

    template <typename Tsolver>
    std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> random_next_outcome(
            Tsolver& solver,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().sample(state, action);
    }

    template <typename Tsolver>
    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            typename Tsolver::ActionNode& action) const {
        return action.dist_to_outcome[action.dist(solver.gen())]->first;
    }
};


/** Default tree policy as used in UCT */
class DefaultTreePolicy {
public :
    template <typename Tsolver, typename Texpander, typename TactionSelector>
    typename Tsolver::StateNode* operator()(Tsolver& solver,
                                            const Texpander& expander,
                                            const TactionSelector& action_selector,
                                            typename Tsolver::StateNode& n,
                                            std::size_t& d) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Launching default tree policy from state " + n.state.print()); }
            solver.transition_mode().init_rollout(solver);
            typename Tsolver::StateNode* current_node = &n;
            while(!(current_node->terminal) && d < solver.max_depth()) {
                typename Tsolver::StateNode* next_node = expander(solver, *current_node);
                d++;
                if (next_node == nullptr) { // node fully expanded
                    typename Tsolver::ActionNode* action = action_selector(solver, *current_node);
                    if (action == nullptr) {
                        throw std::runtime_error("AIRLAPS exception: no best action found in state " + current_node->state.print());
                    } else {
                        next_node = solver.transition_mode().random_next_node(solver, *action);
                        if (next_node == nullptr) { // might happen with step transition mode and stochastic environments
                            break;
                        } else {
                            current_node = next_node;
                        }
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
};


/** Test if a given node needs to be expanded by assuming that applicable actions and next states
 *  can be enumerated. Returns nullptr if all actions and outcomes have already been tried, otherwise a
 *  sampled unvisited outcome according to its probability (among only unvisited outcomes).
 *  REQUIREMENTS: returns nullptr if all actions have been already tried, and set the terminal
 *  flag of the returned next state
 */
class FullExpand {
public :
    FullExpand()
    : _checked_transition_mode(false) {}

    template <typename Tsolver>
    typename Tsolver::StateNode* operator()(Tsolver& solver, typename Tsolver::StateNode& n) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Testing expansion of state " + n.state.print()); }
            if (n.expanded) {
                if (solver.debug_logs()) { spdlog::debug("State already fully expanded"); }
                return nullptr;
            }
            // Generate applicable actions if not already done
            if (n.actions.empty()) {
                if (solver.debug_logs()) { spdlog::debug("State never expanded, generating all next actions"); }
                auto applicable_actions = solver.domain().get_applicable_actions(n.state)->get_elements();
                for (const auto& a : applicable_actions) {
                    auto i = n.actions.emplace(typename Tsolver::ActionNode(a));
                    if (i.second) {
                        // we won't change the real key (ActionNode::action) so we are safe
                        const_cast<typename Tsolver::ActionNode&>(*i.first).parent = &n;
                    }
                }
            }
            // Check for untried outcomes
            if (solver.debug_logs()) { spdlog::debug("Checking for untried outcomes..."); }
            std::vector<std::pair<typename Tsolver::ActionNode*, typename Tsolver::StateNode*>> untried_outcomes;
            std::vector<double> weights;
            for (auto& a : n.actions) {
                // we won't change the real key (ActionNode::action) so we are safe
                typename Tsolver::ActionNode& ca = const_cast<typename Tsolver::ActionNode&>(a);
                if (a.outcomes.empty()) {
                    // we won't change the real key (ActionNode::action) so we are safe
                    untried_outcomes.push_back(std::make_pair(&ca, nullptr));
                    weights.push_back(1.0);
                } else {
                    // Check if there are next states that have been never visited
                    std::vector<double> probs = a.dist.probabilities();
                    for (std::size_t p = 0 ; p < probs.size() ; p++) {
                        typename Tsolver::StateNode* on = ca.dist_to_outcome[p]->first;
                        if (on->visits_count == 0) {
                            untried_outcomes.push_back(std::make_pair(&ca, on));
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
                auto& uo = untried_outcomes[odist(solver.gen())];
                if (uo.second == nullptr) { // unexpanded action
                    if (solver.debug_logs()) { spdlog::debug("Found one unexpanded action: " + uo.first->action.print()); }
                    return expand_action(solver, solver.transition_mode(), n, *(uo.first));
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

    template <typename Tsolver, typename Ttransition_mode,
              std::enable_if_t<std::is_same<Ttransition_mode, DistributionTransitionMode>::value, int> = 0>
    typename Tsolver::StateNode* expand_action(Tsolver& solver, const Ttransition_mode& transition_mode,
                                               typename Tsolver::StateNode& state, typename Tsolver::ActionNode& action) const {
        try {
            // Generate the next states of this action
            auto next_states = solver.domain().get_next_state_distribution(state.state, action.action)->get_values();
            std::vector<typename Tsolver::StateNode*> untried_outcomes;
            std::vector<double> weights;
            std::vector<double> outcome_weights;
            for (const auto& ns : next_states) {
                typename Tsolver::Domain::OutcomeExtractor oe(ns);
                auto i = solver.graph().emplace(oe.state());
                typename Tsolver::StateNode& next_node = const_cast<typename Tsolver::StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                double reward = solver.domain().get_transition_reward(state.state, action.action, next_node.state);
                auto ii = action.outcomes.insert(std::make_pair(&next_node, std::make_pair(reward, 1)));
                if (ii.second) { // new outcome
                    action.dist_to_outcome.push_back(ii.first);
                    outcome_weights.push_back(oe.probability());
                } else { // existing outcome (following code not efficient but hopefully very rare case if domain is well defined)
                    for (unsigned int oid = 0 ; oid < outcome_weights.size() ; oid++) {
                        if (action.dist_to_outcome[oid]->first == ii.first->first) { // found my outcome!
                            std::pair<double, std::size_t>& mp = ii.first->second;
                            mp.first = ((double) (outcome_weights[oid] * mp.first) + (reward * oe.probability())) / ((double) (outcome_weights[oid] + oe.probability()));
                            outcome_weights[oid] += oe.probability();
                            mp.second += 1; // useless in this mode a priori, but just keep track for coherency
                            break;
                        }
                    }
                }
                next_node.parents.push_back(&action);
                if (i.second) { // new node
                    next_node.terminal = solver.domain().is_terminal(next_node.state);
                }
                if (next_node.actions.empty()) {
                    if (solver.debug_logs()) spdlog::debug("Candidate next state: " + next_node.state.print());
                    untried_outcomes.push_back(&next_node);
                    weights.push_back(oe.probability());
                }
            }
            // Record the action's outcomes distribution
            action.dist = std::discrete_distribution<>(outcome_weights.begin(), outcome_weights.end());
            // Pick a random next state
            if (untried_outcomes.empty()) {
                // All next states already visited => pick a random next state using action.dist
                return action.dist_to_outcome[action.dist(solver.gen())]->first;
            } else {
                // Pick a random next state among untried ones
                std::discrete_distribution<> odist(weights.begin(), weights.end());
                return untried_outcomes[odist(solver.gen())];
            }
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when expanding action " + action.action.print() + ": " + e.what());
            throw;
        }
    }

    template <typename Tsolver, typename Ttransition_mode,
              std::enable_if_t<std::is_same<Ttransition_mode, StepTransitionMode>::value ||
                               std::is_same<Ttransition_mode, SampleTransitionMode>::value, int> = 0>
    typename Tsolver::StateNode* expand_action(Tsolver& solver, const Ttransition_mode& transition_mode,
                                               typename Tsolver::StateNode& state, typename Tsolver::ActionNode& action) const {
        try {
            if (!_checked_transition_mode) {
                spdlog::warn("Using MCTS full expansion mode with step() or sample() domain's transition mode assumes the domain is deterministic (unpredictable result otherwise).");
                _checked_transition_mode = true;
            }
            // Generate the next state of this action
            std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> to = transition_mode.random_next_outcome(solver, state.state, action.action);
            auto i = solver.graph().emplace(to->state());
            typename Tsolver::StateNode& next_node = const_cast<typename Tsolver::StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
            auto ii = action.outcomes.insert(std::make_pair(&next_node, std::make_pair(to->reward(), 1)));
            action.dist_to_outcome.push_back(ii.first);
            next_node.parents.push_back(&action);
            if (i.second) { // new node
                next_node.terminal = to->terminal();
            }
            // Record the action's outcomes distribution
            action.dist = std::discrete_distribution<>({1.0});
            if (solver.debug_logs()) spdlog::debug("Candidate next state: " + next_node.state.print());
            return &next_node;
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when expanding action " + action.action.print() + ": " + e.what());
            throw;
        }
    }

private :
    mutable bool _checked_transition_mode;
};


/** Test if a given node needs to be expanded by sampling applicable actions and next states.
 *  Tries to sample new outcomes with a probability proportional to the number of actual expansions.
 *  Returns nullptr if we cannot sample new outcomes, otherwise a sampled unvisited outcome
 *  according to its probability (among only unvisited outcomes).
 *  REQUIREMENTS: returns nullptr if all actions have been already tried, and set the terminal
 *  flag of the returned next state
 */
class PartialExpand {
public :
    template <typename Tsolver>
    typename Tsolver::StateNode* operator()(Tsolver& solver, typename Tsolver::StateNode& n) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Test expansion of state " + n.state.print()); }
            // Sample an action
            std::bernoulli_distribution dist_state_expansion((n.visits_count > 0)?
                                                             (((double) n.expansions_count) / ((double) n.visits_count)):
                                                             1.0);
            typename Tsolver::ActionNode* action_node = nullptr;
            if (dist_state_expansion(solver.gen())) {
                typename Tsolver::Domain::Action action = solver.domain().get_applicable_actions(n.state).sample();
                auto a = n.actions.emplace(typename Tsolver::ActionNode(action));
                if (a.second) { // new action
                    n.expansions_count += 1;
                }
                action_node = &const_cast<typename Tsolver::ActionNode&>(*(a.first)); // we won't change the real key (ActionNode::action) so we are safe
                if (solver.debug_logs()) { spdlog::debug("Tried to sample a new action: " + action_node->action.print()); }
            } else {
                std::vector<typename Tsolver::ActionNode*> actions;
                for (auto& a : n.actions) {
                    actions.push_back(&a);
                }
                std::uniform_int_distribution<> dist_known_actions(0, actions.size()-1);
                action_node = actions[dist_known_actions(solver.gen())];
                if (solver.debug_logs()) { spdlog::debug("Sampled among known actions: " + action_node->action.print()); }
            }
            // Sample an outcome
            std::bernoulli_distribution dist_action_expansion((action_node->visits_count > 0)?
                                                              (((double) action_node->expansions_count) / ((double) action_node->visits_count)):
                                                              1.0);
            typename Tsolver::StateNode* ns = nullptr;
            if (dist_action_expansion(solver.gen())) {
                std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> to = solver.transition_mode().random_next_outcome(n.state, action_node->action);
                auto s = solver.graph().emplace(to->state());
                ns = &const_cast<typename Tsolver::StateNode&>(*(s.first)); // we won't change the real key (StateNode::state) so we are safe
                if (s.second) { // new state
                    ns->terminal = to->termination();
                }
                auto ins = action_node->outcomes.emplace(std::make_pair(ns, std::make_pair(to->reward(), 1)));
                // Update the outcome's reward and visits count
                if (ins.second) { // new outcome
                    action_node->dist_to_outcome.push_back(ins.first);
                    action_node->expansions_count += 1;
                    ns->parents.push_back(action_node);
                } else { // known outcome
                    std::pair<double, std::size_t>& mp = ins.first->second;
                    mp.first = ((double) (mp.second * mp.first) + to->reward()) / ((double) (mp.second + 1));
                    mp.second += 1;
                    ns = nullptr; // we have not discovered anything new
                }
                // Reconstruct the probability distribution
                std::vector<double> weights(action_node->dist_to_outcome.size());
                for (unsigned int oid = 0 ; oid < weights.size() ; oid++) {
                    weights[oid] = action_node->dist_to_outcome[oid]->second.second;
                }
                action_node->dist = std::discrete_distribution<>(weights.begin(), weights.end());
                if (solver.debug_logs()) { spdlog::debug("Tried to sample a new outcome: " + ns->state.print()); }
            } else {
                ns = nullptr; // we have not discovered anything new
                if (solver.debug_logs()) { spdlog::debug("Sampled among known outcomes: " + ns->state.print()); }
            }
            return ns;
        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception in MCTS when expanding state " + n.state.print() + ": " + e.what());
            throw;
        }
    }
};


/** UCB1 Best Child */
class UCB1ActionSelector {
public :
    // 1/sqrt(2) is a good compromise for rewards in [0;1]
    UCB1ActionSelector(double ucb_constant = 1.0 / std::sqrt(2.0))
    : _ucb_constant(ucb_constant) {}

    template <typename Tsolver>
    typename Tsolver::ActionNode* operator()(Tsolver& solver, const typename Tsolver::StateNode& n) const {
        double best_value = -std::numeric_limits<double>::max();
        typename Tsolver::ActionNode* best_action = nullptr;
        for (const auto& a : n.actions) {
            if (a.visits_count > 0) {
                double tentative_value = a.value + (_ucb_constant * std::sqrt((2.0 * std::log(n.visits_count)) / a.visits_count));
                if (tentative_value > best_value) {
                    best_value = tentative_value;
                    best_action = &const_cast<typename Tsolver::ActionNode&>(a); // we won't change the real key (ActionNode::action) so we are safe
                }
            }
        }
        if (solver.debug_logs()) { spdlog::debug("UCB1 selection from state " + n.state.print() +
                                                 ": value=" + std::to_string(best_value) +
                                                 ", action=" + ((best_action != nullptr)?(best_action->action.print()):("nullptr"))); }
        return best_action;
    }

private :
    double _ucb_constant;
};


/** Select action with maximum Q-value */
class BestQValueActionSelector {
public :
    template <typename Tsolver>
    typename Tsolver::ActionNode* operator()(Tsolver& solver, const typename Tsolver::StateNode& n) const {
        double best_value = -std::numeric_limits<double>::max();
        typename Tsolver::ActionNode* best_action = nullptr;
        for (const auto& a : n.actions) {
            if (a.visits_count > 0) {
                if (a.value > best_value) {
                    best_value = a.value;
                    best_action = &const_cast<typename Tsolver::ActionNode&>(a); // we won't change the real key (ActionNode::action) so we are safe
                }
            }
        }
        if (solver.debug_logs()) { spdlog::debug("Best Q-value selection from state " + n.state.print() +
                                                 ": value=" + std::to_string(best_value) +
                                                 ", action=" + ((best_action != nullptr)?(best_action->action.print()):("nullptr"))); }
        return best_action;
    }
};


/** Random default policy */
class RandomDefaultPolicy {
public :
    template <typename Tsolver>
    void operator()(Tsolver& solver, typename Tsolver::StateNode& n, std::size_t d) const {
        try {
            if (solver.debug_logs()) { spdlog::debug("Launching random default policy from state " + n.state.print()); }
            typename Tsolver::Domain::State current_state = n.state;
            bool termination = false;
            std::size_t current_depth = d;
            double reward = 0.0;
            double gamma_n = 1.0;
            while(!termination && current_depth < solver.max_depth()) {
                std::unique_ptr<typename Tsolver::Domain::Action> action = solver.domain().get_applicable_actions(current_state)->sample();

                std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> o = solver.transition_mode().random_next_outcome(solver, current_state, *action);
                reward += gamma_n * (o->reward());
                gamma_n *= solver.discount();
                current_state = o->state();
                termination = o->terminal();
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


/** Graph backup: update Q values using the graph ancestors (rather than only the trajectory leading to n) */
struct GraphBackup {
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
                    double q_value = a->outcomes[f].first + (solver.discount() * (f->value));
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
          typename TexecutionPolicy = ParallelExecution,
          typename TtransitionMode = DistributionTransitionMode,
          typename TtreePolicy = DefaultTreePolicy,
          typename Texpander = FullExpand,
          typename TactionSelectorOptimization = UCB1ActionSelector,
          typename TactionSelectorExecution = BestQValueActionSelector,
          typename TdefaultPolicy = RandomDefaultPolicy,
          typename TbackPropagator = GraphBackup>
class MCTSSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef TtreePolicy TreePolicy;
    typedef Texpander Expander;
    typedef TactionSelectorOptimization ActionSelectorOptimization;
    typedef TactionSelectorExecution ActionSelectorExecution;
    typedef TdefaultPolicy DefaultPolicy;
    typedef TbackPropagator BackPropagator;
    typedef TexecutionPolicy ExecutionPolicy;
    typedef TtransitionMode TransitionMode;

    struct ActionNode;

    struct StateNode {
        typedef typename SetTypeDeducer<ActionNode, Action>::Set ActionSet;
        State state;
        bool terminal;
        bool expanded; // used only for full expansion mode
        std::size_t expansions_count; // used only for partial expansion mode
        ActionSet actions;
        double value;
        std::size_t visits_count;
        std::list<ActionNode*> parents;

        StateNode(const State& s)
            : state(s), terminal(false), expanded(false),
              expansions_count(0), value(0.0), visits_count(0) {}
        
        struct Key {
            const State& operator()(const StateNode& sn) const { return sn.state; }
        };
    };

    struct ActionNode {
        Action action;
        std::unordered_map<StateNode*, std::pair<double, std::size_t>> outcomes; // next state nodes owned by _graph
        std::vector<typename std::unordered_map<StateNode*, std::pair<double, std::size_t>>::iterator> dist_to_outcome;
        std::discrete_distribution<> dist;
        std::size_t expansions_count; // used only for partial expansion mode
        double value;
        std::size_t visits_count;
        StateNode* parent;

        ActionNode(const Action& a)
            : action(a), expansions_count(0), value(0.0),
              visits_count(0), parent(nullptr) {}
        
        struct Key {
            const Action& operator()(const ActionNode& an) const { return an.action; }
        };
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
               const ActionSelectorOptimization& action_selector_optimization = ActionSelectorOptimization(),
               const ActionSelectorExecution& action_selector_execution = ActionSelectorExecution(),
               const DefaultPolicy& default_policy = DefaultPolicy(),
               const BackPropagator& back_propagator = BackPropagator())
    : _domain(domain),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _discount(discount), _nb_rollouts(0),
      _debug_logs(debug_logs), _tree_policy(tree_policy), _expander(expander),
      _action_selector_optimization(action_selector_optimization),
      _action_selector_execution(action_selector_execution),
      _default_policy(default_policy), _back_propagator(back_propagator) {
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
                StateNode* sn = _tree_policy(*this, _expander, _action_selector_optimization, root_node, depth);
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
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            return false;
        } else {
            return _action_selector_execution(*this, *si) != nullptr;
        }
    }

    Action get_best_action(const State& s) {
        auto si = _graph.find(s);
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = _action_selector_execution(*this, *si);
        }
        if (action == nullptr) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        } else {
            if (_debug_logs) {
                std::string str = "(";
                for (const auto& o : action->outcomes) {
                    str += "\n    " + o.first->state.print();
                }
                str += "\n)";
                spdlog::debug("Best action's known outcomes:\n" + str);
            }
            _action_prefix.push_back(action->action);
            return action->action;
        }
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(StateNode(s));
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = _action_selector_execution(*this, *si);
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

    typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() {
        typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
        for (auto& n : _graph) {
            ActionNode* action = _action_selector_execution(*this, n);
            if (action != nullptr) {
                p.insert(std::make_pair(n.state, std::make_pair(action->action, action->value)));
            }
        }
        return p;
    }

    Domain& domain() { return _domain; }

    std::size_t time_budget() const { return _time_budget; }

    std::size_t rollout_budget() const { return _rollout_budget; }

    std::size_t max_depth() const { return _max_depth; }

    double discount() const { return _discount; }

    TransitionMode& transition_mode() { return _transition_mode; }

    const TreePolicy& tree_policy() { return _tree_policy; }

    const Expander& expander() { return _expander; }

    const ActionSelectorOptimization& action_selector_optimization() { return _action_selector_optimization; }

    const ActionSelectorExecution& action_selector_execution() { return _action_selector_execution; }

    const DefaultPolicy& default_policy() { return _default_policy; }

    const BackPropagator& back_propagator() { return _back_propagator; }

    Graph& graph() { return _graph; }

    const std::list<Action>& action_prefix() const { return _action_prefix; }

    std::mt19937& gen() { return *_gen; }

    bool debug_logs() const { return _debug_logs; }

private :

    Domain& _domain;
    std::size_t _time_budget;
    std::size_t _rollout_budget;
    std::size_t _max_depth;
    double _discount;
    std::size_t _nb_rollouts;
    bool _debug_logs;
    TransitionMode _transition_mode;
    TreePolicy _tree_policy;
    Expander _expander;
    ActionSelectorOptimization _action_selector_optimization;
    ActionSelectorExecution _action_selector_execution;
    DefaultPolicy _default_policy;
    BackPropagator _back_propagator;

    Graph _graph;
    std::list<Action> _action_prefix;

    std::unique_ptr<std::mt19937> _gen;
}; // MCTSSolver class

/** UCT is MCTS with the default template options */
template <typename Tdomain, typename Texecution_policy, typename TtransitionMode, typename ...T>
using UCTSolver = MCTSSolver<Tdomain, Texecution_policy, TtransitionMode, T...>;

} // namespace airlaps

#endif // AIRLAPS_MCTS_HH
