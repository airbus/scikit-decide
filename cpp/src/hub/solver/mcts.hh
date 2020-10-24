/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the scikit-decide implementation of MCTS and UCT from
 * "A Survey of Monte Carlo Tree Search Methods" by Browne et al
 * (IEEE Transactions on Computational Intelligence  and AI in games,
 * 2012). We additionnally implement a heuristic value estimate as in
 * "Monte-Carlo tree search and rapid action value estimation in 
 * computer Go" by Gelly and Silver (Artificial Intelligence, 2011)
 * except that the heuristic estimate is called on states but not
 * on state-action pairs to be more in line with heuristic search
 * algorithms in the literature and other implementations of
 * heuristic search algorithms in scikit-decide.
 */
#ifndef SKDECIDE_MCTS_HH
#define SKDECIDE_MCTS_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include <boost/range/irange.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"

namespace skdecide {

/** Use Environment domain knowledge for transitions */
template <typename Tsolver>
struct StepTransitionMode {
    void init_rollout(Tsolver& solver, const std::size_t* thread_id) const {
        solver.domain().reset(thread_id);
        std::for_each(solver.action_prefix().begin(), solver.action_prefix().end(),
                      [&solver, &thread_id](const typename Tsolver::Domain::Event& a){solver.domain().step(a, thread_id);});
    }

    typename Tsolver::Domain::TransitionOutcome random_next_outcome(
            Tsolver& solver,
            const std::size_t* thread_id,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().step(action, thread_id);
    }

    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            const std::size_t* thread_id,
            typename Tsolver::ActionNode& action) const {
        auto outcome = solver.domain().step(action.action, thread_id);
        typename Tsolver::StateNode* n = nullptr;

        solver.execution_policy().protect([&n, &solver, &outcome](){
            auto si = solver.graph().find(typename Tsolver::StateNode(outcome.state()));
            if (si != solver.graph().end()) {
                // we won't change the real key (ActionNode::action) so we are safe
                n = &const_cast<typename Tsolver::StateNode&>(*si);
            }
        });

        return n;
    }
};


/** Use Simulation domain knowledge for transitions */
template <typename Tsolver>
struct SampleTransitionMode {
    void init_rollout(Tsolver& solver, const std::size_t* thread_id) const {}

    typename Tsolver::Domain::TransitionOutcome random_next_outcome(
            Tsolver& solver,
            const std::size_t* thread_id,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().sample(state, action, thread_id);
    }

    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            const std::size_t* thread_id,
            typename Tsolver::ActionNode& action) const {
        typename Tsolver::StateNode* n = nullptr;

        solver.execution_policy().protect([&n, &action, &solver](){
            solver.execution_policy().protect([&n, &action, &solver](){
                n = action.dist_to_outcome[action.dist(solver.gen())]->first;
            }, solver.gen_mutex());
        }, action.parent->mutex);

        return n;
    }
};


/** Use uncertain transitions domain knowledge for transitions */
template <typename Tsolver>
struct DistributionTransitionMode {
    void init_rollout(Tsolver& solver, const std::size_t* thread_id) const {}

    typename Tsolver::Domain::TransitionOutcome random_next_outcome(
            Tsolver& solver,
            const std::size_t* thread_id,
            const typename Tsolver::Domain::State& state,
            const typename Tsolver::Domain::Event& action) const {
        return solver.domain().sample(state, action, thread_id);
    }

    typename Tsolver::StateNode* random_next_node(
            Tsolver& solver,
            const std::size_t* thread_id,
            typename Tsolver::ActionNode& action) const {
        typename Tsolver::StateNode* n = nullptr;

        solver.execution_policy().protect([&n, &action, &solver](){
            solver.execution_policy().protect([&n, &action, &solver](){
                n = action.dist_to_outcome[action.dist(solver.gen())]->first;
            }, solver.gen_mutex());
        }, action.parent->mutex);

        return n;
    }
};


/** Default tree policy as used in UCT */
template <typename Tsolver>
class DefaultTreePolicy {
public :
    typename Tsolver::StateNode* operator()(Tsolver& solver,
                                            const std::size_t* thread_id, // for parallelisation
                                            const typename Tsolver::Expander& expander,
                                            const typename Tsolver::ActionSelectorOptimization& action_selector,
                                            typename Tsolver::StateNode& n,
                                            std::size_t& d) const {
        try {
            if (solver.debug_logs()) {
                solver.execution_policy().protect([&n](){
                    spdlog::debug("Launching default tree policy from state " + n.state.print() +
                                  Tsolver::ExecutionPolicy::print_thread());
                }, n.mutex);
            }

            solver.transition_mode().init_rollout(solver, thread_id);
            typename Tsolver::StateNode* current_node = &n;

            while(!(current_node->terminal) && d < solver.max_depth()) {
                typename Tsolver::StateNode* next_node = expander(solver, thread_id, *current_node);
                d++;

                if (next_node == nullptr) { // node fully expanded
                    typename Tsolver::ActionNode* action = action_selector(solver, thread_id, *current_node);
                    
                    if (action == nullptr) {
                        // It might happen in parallel execution mode when the current node's actions are all being
                        // expanded by concurrent threads that claim the node is expanded but not yet backpropagated
                        // and a new thread meantime comes and sees the node as expanded, thus all action visits counts
                        // are still equal to zero (implying action_selector to return nullptr).
                        // This shall NOT happen in sequential execution mode.
                        break;
                    } else {
                        next_node = solver.transition_mode().random_next_node(solver, thread_id, *action);

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
            solver.execution_policy().protect([&n, &e](){
                spdlog::error("SKDECIDE exception in MCTS when simulating the tree policy from state " + n.state.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, n.mutex);
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
template <typename Tsolver>
class FullExpand {
public :
    typedef std::function<std::pair<double, std::size_t>
                          (typename Tsolver::Domain&, const typename Tsolver::Domain::State&, const std::size_t*)> HeuristicFunctor;

    FullExpand(const HeuristicFunctor& heuristic = [](typename Tsolver::Domain& domain,
                                                      const typename Tsolver::Domain::State& state,
                                                      const std::size_t* thread_id) {
        return std::make_pair(0.0, 0);
    })
    : _heuristic(heuristic), _checked_transition_mode(false) {}

    FullExpand(const FullExpand& other)
    : _heuristic(other._heuristic), _checked_transition_mode(false) {}

    typename Tsolver::StateNode* operator()(Tsolver& solver,
                                            const std::size_t* thread_id,
                                            typename Tsolver::StateNode& n) const {
        try {
            if (solver.debug_logs()) {
                solver.execution_policy().protect([&n](){
                    spdlog::debug("Testing expansion of state " + n.state.print() +
                                  Tsolver::ExecutionPolicy::print_thread());
                }, n.mutex);
            }

            if (n.expanded) {
                if (solver.debug_logs()) { spdlog::debug("State already fully expanded" +
                                                         Tsolver::ExecutionPolicy::print_thread()); }
                return nullptr;
            }

            // Generate applicable actions if not already done
            solver.execution_policy().protect([&n, &solver, &thread_id](){
                if (n.actions.empty()) {
                    if (solver.debug_logs()) { spdlog::debug("State never expanded, generating all next actions" +
                                                             Tsolver::ExecutionPolicy::print_thread()); }
                    auto applicable_actions = solver.domain().get_applicable_actions(n.state, thread_id).get_elements();

                    for (const auto& a : applicable_actions) {
                        auto i = n.actions.emplace(typename Tsolver::ActionNode(a));

                        if (i.second) {
                            // we won't change the real key (ActionNode::action) so we are safe
                            const_cast<typename Tsolver::ActionNode&>(*i.first).parent = &n;
                        }
                    }
                }
            }, n.mutex);

            // Check for untried outcomes
            if (solver.debug_logs()) { spdlog::debug("Checking for untried outcomes" +
                                                     Tsolver::ExecutionPolicy::print_thread()); }
            std::vector<std::pair<typename Tsolver::ActionNode*, typename Tsolver::StateNode*>> untried_outcomes;
            std::vector<double> weights;

            solver.execution_policy().protect([&n, &untried_outcomes, &weights](){
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
            }, n.mutex);

            if (untried_outcomes.empty()) { // nothing to expand
                if (solver.debug_logs()) { spdlog::debug("All outcomes already tried" +
                                                         Tsolver::ExecutionPolicy::print_thread()); }
                n.expanded = true;
                return nullptr;
            } else {
                std::discrete_distribution<> odist(weights.begin(), weights.end());
                std::size_t outcome_id = 0;
                solver.execution_policy().protect([&outcome_id, &odist, &solver](){
                    outcome_id = odist(solver.gen());
                }, solver.gen_mutex());
                auto& uo = untried_outcomes[outcome_id];

                if (uo.second == nullptr) { // unexpanded action
                    if (solver.debug_logs()) { spdlog::debug("Found one unexpanded action: " + uo.first->action.print() +
                                                             Tsolver::ExecutionPolicy::print_thread()); }
                    return expand_action(solver, thread_id, solver.transition_mode(), n, *(uo.first));
                } else { // expanded action, just return the selected next state
                    if (solver.debug_logs()) {
                        solver.execution_policy().protect([&uo](){
                            spdlog::debug("Found one untried outcome: action " + uo.first->action.print() +
                                          " and next state " + uo.second->state.print() +
                                          Tsolver::ExecutionPolicy::print_thread());
                        }, uo.second->mutex);
                    }
                    return uo.second;
                }
            }
        } catch (const std::exception& e) {
            solver.execution_policy().protect([&n, &e](){
                spdlog::error("SKDECIDE exception in MCTS when expanding state " + n.state.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, n.mutex);
            throw;
        }
    }

    template <typename Ttransition_mode,
              std::enable_if_t<std::is_same<Ttransition_mode, DistributionTransitionMode<Tsolver>>::value, int> = 0>
    typename Tsolver::StateNode* expand_action(Tsolver& solver,
                                               const std::size_t* thread_id,
                                               const Ttransition_mode& transition_mode,
                                               typename Tsolver::StateNode& state,
                                               typename Tsolver::ActionNode& action) const {
        try {
            // Generate the next states of this action
            auto next_states = solver.domain().get_next_state_distribution(state.state, action.action, thread_id).get_values();
            std::vector<typename Tsolver::StateNode*> untried_outcomes;
            std::vector<double> weights;
            std::vector<double> outcome_weights;

            for (auto ns : next_states) {
                std::pair<typename Tsolver::Graph::iterator, bool> i;

                solver.execution_policy().protect([&i, &solver, &ns](){
                    i = solver.graph().emplace(ns.state());
                });

                typename Tsolver::StateNode& next_node = const_cast<typename Tsolver::StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                double reward = 0.0;

                solver.execution_policy().protect([&reward, &solver, &state, &action, &next_node, &thread_id](){
                    reward = solver.domain().get_transition_reward(state.state, action.action, next_node.state, thread_id);
                }, next_node.mutex);

                solver.execution_policy().protect([&action, &next_node, &outcome_weights, &reward, &ns](){
                    auto ii = action.outcomes.insert(std::make_pair(&next_node, std::make_pair(reward, 1)));

                    if (ii.second) { // new outcome
                        action.dist_to_outcome.push_back(ii.first);
                        outcome_weights.push_back(ns.probability());
                    } else { // existing outcome (following code not efficient but hopefully very rare case if domain is well defined)
                        for (unsigned int oid = 0 ; oid < outcome_weights.size() ; oid++) {
                            if (action.dist_to_outcome[oid]->first == ii.first->first) { // found my outcome!
                                std::pair<double, std::size_t>& mp = ii.first->second;
                                mp.first = ((double) (outcome_weights[oid] * mp.first) + (reward * ns.probability())) / ((double) (outcome_weights[oid] + ns.probability()));
                                outcome_weights[oid] += ns.probability();
                                mp.second += 1; // useless in this mode a priori, but just keep track for coherency
                                break;
                            }
                        }
                    }
                }, action.parent->mutex);

                solver.execution_policy().protect([this, &next_node, &action, &i, &solver, &thread_id, &untried_outcomes, &weights, &ns](){
                    next_node.parents.insert(&action);

                    if (i.second) { // new node
                        next_node.terminal = solver.domain().is_terminal(next_node.state, thread_id);
                        std::pair<double, std::size_t> h = _heuristic(solver.domain(), next_node.state, thread_id);
                        next_node.value = h.first;
                        next_node.visits_count = h.second;
                    }

                    if (next_node.actions.empty()) {
                        if (solver.debug_logs()) spdlog::debug("Candidate next state: " + next_node.state.print() +
                                                               Tsolver::ExecutionPolicy::print_thread());
                        untried_outcomes.push_back(&next_node);
                        weights.push_back(ns.probability());
                    }
                }, next_node.mutex);
            }

            // Record the action's outcomes distribution
            solver.execution_policy().protect([&action, &outcome_weights](){
                action.dist = std::discrete_distribution<>(outcome_weights.begin(), outcome_weights.end());
            }, action.parent->mutex);

            // Pick a random next state
            if (untried_outcomes.empty()) {
                // All next states already visited => pick a random next state using action.dist
                std::size_t outcome_id = 0;

                solver.execution_policy().protect([&outcome_id, &action, &solver](){
                    outcome_id = action.dist(solver.gen());
                }, solver.gen_mutex());

                typename Tsolver::StateNode* outcome = nullptr;

                solver.execution_policy().protect([&action, &outcome, &outcome_id](){
                    outcome = action.dist_to_outcome[outcome_id]->first;
                }, action.parent->mutex);

                return outcome;
            } else {
                // Pick a random next state among untried ones
                std::discrete_distribution<> odist(weights.begin(), weights.end());
                std::size_t outcome_id = 0;

                solver.execution_policy().protect([&outcome_id, &odist, &solver](){
                    outcome_id = odist(solver.gen());
                }, solver.gen_mutex());

                return untried_outcomes[outcome_id];
            }
        } catch (const std::exception& e) {
            solver.execution_policy().protect([&action, &e](){
                spdlog::error("SKDECIDE exception in MCTS when expanding action " + action.action.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, action.parent->mutex);
            throw;
        }
    }

    template <typename Ttransition_mode,
              std::enable_if_t<std::is_same<Ttransition_mode, StepTransitionMode<Tsolver>>::value ||
                               std::is_same<Ttransition_mode, SampleTransitionMode<Tsolver>>::value, int> = 0>
    typename Tsolver::StateNode* expand_action(Tsolver& solver,
                                               const std::size_t* thread_id,
                                               const Ttransition_mode& transition_mode,
                                               typename Tsolver::StateNode& state,
                                               typename Tsolver::ActionNode& action) const {
        try {
            if (!_checked_transition_mode) {
                spdlog::warn("Using MCTS full expansion mode with step() or sample() domain's transition mode assumes the domain is deterministic (unpredictable result otherwise).");
                _checked_transition_mode = true;
            }
            // Generate the next state of this action
            typename Tsolver::Domain::TransitionOutcome to = transition_mode.random_next_outcome(solver, thread_id, state.state, action.action);
            std::pair<typename Tsolver::Graph::iterator, bool> i;

            solver.execution_policy().protect([&i, &solver, &to](){
                i = solver.graph().emplace(to.state());
            });
            
            typename Tsolver::StateNode& next_node = const_cast<typename Tsolver::StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
            
            solver.execution_policy().protect([&action, &next_node, &to](){
                auto ii = action.outcomes.insert(std::make_pair(&next_node, std::make_pair(to.reward(), 1)));
                action.dist_to_outcome.push_back(ii.first);
            }, action.parent->mutex);

            solver.execution_policy().protect([this, &next_node, &action, &i, &to, &solver, &thread_id](){
                next_node.parents.insert(&action);

                if (i.second) { // new node
                    next_node.terminal = to.terminal();
                    std::pair<double, std::size_t> h = _heuristic(solver.domain(), next_node.state, thread_id);
                    next_node.value = h.first;
                    next_node.visits_count = h.second;
                }
            }, next_node.mutex);

            // Record the action's outcomes distribution
            solver.execution_policy().protect([&action](){
                action.dist = std::discrete_distribution<>({1.0});
            }, action.parent->mutex);

            if (solver.debug_logs()) {
                solver.execution_policy().protect([&next_node](){
                    spdlog::debug("Candidate next state: " + next_node.state.print() +
                                  Tsolver::ExecutionPolicy::print_thread());
                }, next_node.mutex);
            }
            
            return &next_node;
        } catch (const std::exception& e) {
            solver.execution_policy().protect([&action, &e](){
                spdlog::error("SKDECIDE exception in MCTS when expanding action " + action.action.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, action.parent->mutex);
            throw;
        }
    }

private :
    HeuristicFunctor _heuristic;
    mutable typename Tsolver::ExecutionPolicy::template atomic<bool> _checked_transition_mode;
};


/** Test if a given node needs to be expanded by sampling applicable actions and next states.
 *  Tries to sample new outcomes with a probability proportional to the number of actual expansions.
 *  Returns nullptr if we cannot sample new outcomes, otherwise a sampled unvisited outcome
 *  according to its probability (among only unvisited outcomes).
 *  REQUIREMENTS: returns nullptr if all actions have been already tried, and set the terminal
 *  flag of the returned next state
 */
template <typename Tsolver>
class PartialExpand {
public :
    typedef std::function<std::pair<double, std::size_t>
                          (typename Tsolver::Domain&, const typename Tsolver::Domain::State&, const std::size_t*)> HeuristicFunctor;

    PartialExpand(const HeuristicFunctor& heuristic = [](typename Tsolver::Domain& domain,
                                                         const typename Tsolver::Domain::State& state,
                                                         const std::size_t* thread_id) {
        return std::make_pair(0.0, 0);
    })
    : _heuristic(heuristic) {}

    PartialExpand(const PartialExpand& other)
    : _heuristic(other._heuristic) {}

    typename Tsolver::StateNode* operator()(Tsolver& solver,
                                            const std::size_t* thread_id,
                                            typename Tsolver::StateNode& n) const {
        try {
            if (solver.debug_logs()) {
                solver.execution_policy().protect([&n](){
                    spdlog::debug("Test expansion of state " + n.state.print() +
                                  Tsolver::ExecutionPolicy::print_thread());
                }, n.mutex);
            }

            // Sample an action
            std::bernoulli_distribution dist_state_expansion((n.visits_count > 0)?
                                                             (((double) n.expansions_count) / ((double) n.visits_count)):
                                                             1.0);
            typename Tsolver::ActionNode* action_node = nullptr;
            bool dist_res = false;

            solver.execution_policy().protect([&dist_res, &solver, &dist_state_expansion](){
                dist_res = dist_state_expansion(solver.gen());
            }, solver.gen_mutex());

            if (dist_res) {
                typename Tsolver::Domain::Action action = solver.domain().get_applicable_actions(n.state, thread_id).sample();
                solver.execution_policy().protect([&n, &action, &action_node, &solver](){
                    auto a = n.actions.emplace(typename Tsolver::ActionNode(action));

                    if (a.second) { // new action
                        n.expansions_count += 1;
                    }

                    action_node = &const_cast<typename Tsolver::ActionNode&>(*(a.first)); // we won't change the real key (ActionNode::action) so we are safe
                    if (solver.debug_logs()) { spdlog::debug("Tried to sample a new action: " + action_node->action.print() +
                                                             Tsolver::ExecutionPolicy::print_thread()); }
                }, n.mutex);
            } else {
                std::vector<typename Tsolver::ActionNode*> actions;

                solver.execution_policy().protect([&n, &actions](){
                    for (auto& a : n.actions) {
                        actions.push_back(&a);
                    }
                }, n.mutex);

                std::uniform_int_distribution<> dist_known_actions(0, actions.size()-1);
                std::size_t action_id = 0;

                solver.execution_policy().protect([&action_id, &solver, &dist_known_actions](){
                    action_id = dist_known_actions(solver.gen());
                }, solver.gen_mutex());

                action_node = actions[action_id];
                if (solver.debug_logs()) {
                    solver.execution_policy().protect([&action_node](){
                        spdlog::debug("Sampled among known actions: " + action_node->action.print() +
                                      Tsolver::ExecutionPolicy::print_thread());
                    }, action_node->parent.mutex);
                }
            }

            // Sample an outcome
            std::bernoulli_distribution dist_action_expansion((action_node->visits_count > 0)?
                                                              (((double) action_node->expansions_count) / ((double) action_node->visits_count)):
                                                              1.0);
            typename Tsolver::StateNode* ns = nullptr;
            
            solver.execution_policy().protect([&dist_res, &solver, &dist_action_expansion](){
                dist_res = dist_action_expansion(solver.gen());
            }, solver.gen_mutex());

            if (dist_res) {
                std::unique_ptr<typename Tsolver::Domain::TransitionOutcome> to = solver.transition_mode().random_next_outcome(solver, thread_id, n.state, action_node->action);
                std::pair<typename Tsolver::Graph::iterator, bool> s;

                solver.execution_policy().protect([&s, &solver, &to](){
                    s = solver.graph().emplace(to->state());
                });
                
                ns = &const_cast<typename Tsolver::StateNode&>(*(s.first)); // we won't change the real key (StateNode::state) so we are safe

                if (s.second) { // new state
                    solver.execution_policy().protect([this, &ns, &to, &solver, &thread_id](){
                        ns->terminal = to->termination();
                        std::pair<double, std::size_t> h = _heuristic(solver.domain(), ns->state, thread_id);
                        ns->value = h.first;
                        ns->visits_count = h.second;
                    }, ns->mutex);
                }

                std::pair<typename Tsolver::ActionNode::OutcomeMap::iterator, bool> ins;
                solver.execution_policy().protect([&action_node, &ns, &to, ins](){
                    ins = action_node->outcomes.emplace(std::make_pair(ns, std::make_pair(to->reward(), 1)));
                }, action_node->parent->mutex);

                // Update the outcome's reward and visits count
                if (ins.second) { // new outcome
                    solver.execution_policy().protect([&action_node, &ins](){
                        action_node->dist_to_outcome.push_back(ins.first);
                        action_node->expansions_count += 1;
                    }, action_node->parent->mutex);

                    solver.execution_policy().protect([&ns, &action_node](){
                        ns->parents.insert(action_node);
                    }, ns->mutex);
                } else { // known outcome
                    solver.execution_policy().protect([&ins, &to, &ns](){
                        std::pair<double, std::size_t>& mp = ins.first->second;
                        mp.first = ((double) (mp.second * mp.first) + to->reward()) / ((double) (mp.second + 1));
                        mp.second += 1;
                        ns = nullptr; // we have not discovered anything new
                    }, action_node->parent->mutex);
                }

                // Reconstruct the probability distribution
                solver.execution_policy().protect([&action_node](){
                    std::vector<double> weights(action_node->dist_to_outcome.size());

                    for (unsigned int oid = 0 ; oid < weights.size() ; oid++) {
                        weights[oid] = action_node->dist_to_outcome[oid]->second.second;
                    }

                    action_node->dist = std::discrete_distribution<>(weights.begin(), weights.end());
                }, action_node->parent->mutex);

                if (solver.debug_logs()) {
                    solver.execution_policy().protect([&ns](){
                        spdlog::debug("Tried to sample a new outcome: " + ns->state.print() +
                                      Tsolver::ExecutionPolicy::print_thread());
                    });
                }
            } else {
                ns = nullptr; // we have not discovered anything new

                if (solver.debug_logs()) {
                    solver.execution_policy().protect([&ns](){
                        spdlog::debug("Sampled among known outcomes: " + ns->state.print() +
                                      Tsolver::ExecutionPolicy::print_thread());
                    });
                }
            }

            return ns;
        } catch (const std::exception& e) {
            solver.execution_policy().protect([&n, &e](){
                spdlog::error("SKDECIDE exception in MCTS when expanding state " + n.state.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, n.mutex);
            throw;
        }
    }

private :
    HeuristicFunctor _heuristic;
};


/** UCB1 Best Child */
template <typename Tsolver>
class UCB1ActionSelector {
public :
    // 1/sqrt(2) is a good compromise for rewards in [0;1]
    UCB1ActionSelector(double ucb_constant = 1.0 / std::sqrt(2.0))
    : _ucb_constant(ucb_constant) {}

    UCB1ActionSelector(const UCB1ActionSelector& other)
    : _ucb_constant((double) other._ucb_constant) {}

    typename Tsolver::ActionNode* operator()(Tsolver& solver,
                                             const std::size_t* thread_id,
                                             const typename Tsolver::StateNode& n) const {
        double best_value = -std::numeric_limits<double>::max();
        typename Tsolver::ActionNode* best_action = nullptr;

        solver.execution_policy().protect([this, &n, &best_value, &best_action, &solver](){
            for (const auto& a : n.actions) {
                if (a.visits_count > 0) {
                    double tentative_value = a.value + (2.0 * _ucb_constant * std::sqrt((2.0 * std::log((double) n.visits_count)) / ((double) a.visits_count)));

                    if (tentative_value > best_value) {
                        best_value = tentative_value;
                        best_action = &const_cast<typename Tsolver::ActionNode&>(a); // we won't change the real key (ActionNode::action) so we are safe
                    }
                }
            }

            if (solver.debug_logs()) { spdlog::debug("UCB1 selection from state " + n.state.print() +
                                                     ": value=" + StringConverter::from(best_value) +
                                                     ", action=" + ((best_action != nullptr)?(best_action->action.print()):("nullptr")) +
                                                     Tsolver::ExecutionPolicy::print_thread()); }
        }, n.mutex);
        
        return best_action;
    }

private :
    typename Tsolver::ExecutionPolicy::template atomic<double> _ucb_constant;
};


/** Select action with maximum Q-value */
template <typename Tsolver>
class BestQValueActionSelector {
public :
    typename Tsolver::ActionNode* operator()(Tsolver& solver,
                                             const std::size_t* thread_id,
                                             const typename Tsolver::StateNode& n) const {
        double best_value = -std::numeric_limits<double>::max();
        typename Tsolver::ActionNode* best_action = nullptr;

        solver.execution_policy().protect([&n, &best_value, &best_action, &solver](){
            for (const auto& a : n.actions) {
                if (a.visits_count > 0) {
                    if (a.value > best_value) {
                        best_value = a.value;
                        best_action = &const_cast<typename Tsolver::ActionNode&>(a); // we won't change the real key (ActionNode::action) so we are safe
                    }
                }
            }

            if (solver.debug_logs()) { spdlog::debug("Best Q-value selection from state " + n.state.print() +
                                                     ": value=" + StringConverter::from(best_value) +
                                                     ", action=" + ((best_action != nullptr)?(best_action->action.print()):("nullptr")) +
                                                     Tsolver::ExecutionPolicy::print_thread()); }
        }, n.mutex);
        
        return best_action;
    }
};


/** Default rollout policy */
template <typename Tsolver>
class DefaultRolloutPolicy {
public :
    typedef std::function<typename Tsolver::Domain::Action
                          (typename Tsolver::Domain&, const typename Tsolver::Domain::State&, const std::size_t*)> PolicyFunctor;

    DefaultRolloutPolicy(const PolicyFunctor& policy = [](typename Tsolver::Domain& domain,
                                                          const typename Tsolver::Domain::State& state,
                                                          const std::size_t* thread_id) {
        return domain.get_applicable_actions(state, thread_id).sample();
    })
    : _policy(policy) {}

    void operator()(Tsolver& solver,
                    const std::size_t* thread_id,
                    typename Tsolver::StateNode& n,
                    std::size_t d) const {
        try {
            typename Tsolver::Domain::State current_state;

            solver.execution_policy().protect([&solver, &n, &current_state](){
                if (solver.debug_logs()) { spdlog::debug("Launching default rollout policy from state " + n.state.print() +
                                                         Tsolver::ExecutionPolicy::print_thread()); }
                current_state = n.state;
            }, n.mutex);

            bool termination = false;
            std::size_t current_depth = d;
            double reward = 0.0;
            double gamma_n = 1.0;

            while(!termination && current_depth < solver.max_depth()) {
                typename Tsolver::Domain::Action action = _policy(solver.domain(), current_state, thread_id);
                typename Tsolver::Domain::TransitionOutcome o = solver.transition_mode().random_next_outcome(solver, thread_id, current_state, action);
                reward += gamma_n * (o.reward());
                gamma_n *= solver.discount();
                current_state = o.state();
                termination = o.terminal();
                current_depth++;
                if (solver.debug_logs()) { spdlog::debug("Sampled transition: action=" + action.print() +
                                                         ", next state=" + current_state.print() +
                                                         ", reward=" + StringConverter::from(o.reward()) +
                                                         Tsolver::ExecutionPolicy::print_thread()); }
            }

            // since we can come to state n after exhausting the depth, n might be already visited
            // so don't erase its value but rather update it
            solver.execution_policy().protect([&n, &reward](){
                n.value = ((n.visits_count * n.value)  + reward) / ((double) (n.visits_count + 1));
                n.visits_count += 1;
            }, n.mutex);
        } catch (const std::exception& e) {
            solver.execution_policy().protect([&n, &e](){
                spdlog::error("SKDECIDE exception in MCTS when simulating the random default policy from state " + n.state.print() + ": " + e.what() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, n.mutex);
            throw;
        }
    }

private :
    PolicyFunctor _policy;
};


/** Graph backup: update Q values using the graph ancestors (rather than only the trajectory leading to n) */
template <typename Tsolver>
struct GraphBackup {
    void operator()(Tsolver& solver,
                    const std::size_t* thread_id,
                    typename Tsolver::StateNode& n) const {
        if (solver.debug_logs()) {
            solver.execution_policy().protect([&n](){
                spdlog::debug("Back-propagating values from state " + n.state.print() +
                              Tsolver::ExecutionPolicy::print_thread());
            }, n.mutex);
        }

        std::size_t depth = 0; // used to prevent infinite loop in case of cycles
        std::unordered_set<typename Tsolver::StateNode*> frontier;
        frontier.insert(&n);

        while (!frontier.empty() && depth <= solver.max_depth()) {
            depth++;
            std::unordered_set<typename Tsolver::StateNode*> new_frontier;
            
            for (auto& f : frontier) {
                update_frontier(solver, new_frontier, f, &solver.execution_policy());
            }

            frontier = new_frontier;
        }
    }

    template <typename Texecution_policy,
              std::enable_if_t<std::is_same<Texecution_policy, SequentialExecution>::value, int> = 0>
    static void update_frontier(Tsolver& solver,
                                std::unordered_set<typename Tsolver::StateNode*>& new_frontier, typename Tsolver::StateNode* f,
                                [[maybe_unused]] Texecution_policy* execution_policy) {
        for (auto& a : f->parents) {
            double q_value = a->outcomes[f].first + (solver.discount() * (f->value));
            a->value = (((a->visits_count) * (a->value))  + q_value) / ((double) (a->visits_count + 1));
            a->visits_count += 1;
            typename Tsolver::StateNode* parent_node = a->parent;
            parent_node->value = (((parent_node->visits_count) * (parent_node->value))  + (a->value)) / ((double) (parent_node->visits_count + 1));
            parent_node->visits_count += 1;
            new_frontier.insert(parent_node);
            if (solver.debug_logs()) { spdlog::debug("Updating state " + parent_node->state.print() +
                                                    ": value=" + StringConverter::from(parent_node->value) +
                                                    ", visits=" + StringConverter::from(parent_node->visits_count) +
                                                    Tsolver::ExecutionPolicy::print_thread()); }
        }
    }

    template <typename Texecution_policy,
              std::enable_if_t<std::is_same<Texecution_policy, ParallelExecution>::value, int> = 0>
    static void update_frontier(Tsolver& solver,
                                std::unordered_set<typename Tsolver::StateNode*>& new_frontier, typename Tsolver::StateNode* f,
                                [[maybe_unused]] Texecution_policy* execution_policy) {
        std::list<typename Tsolver::ActionNode*> parents;
        solver.execution_policy().protect([&f, &parents](){
            std::copy(f->parents.begin(), f->parents.end(), std::inserter(parents, parents.end()));
        }, f->mutex);
        for (auto& a : parents) {
            solver.execution_policy().protect([&a, &solver, &f, &new_frontier](){
                double q_value = a->outcomes[f].first + (solver.discount() * (f->value));
                a->value = (((a->visits_count) * (a->value))  + q_value) / ((double) (a->visits_count + 1));
                a->visits_count += 1;
                typename Tsolver::StateNode* parent_node = a->parent;
                parent_node->value = (((parent_node->visits_count) * (parent_node->value))  + (a->value)) / ((double) (parent_node->visits_count + 1));
                parent_node->visits_count += 1;
                new_frontier.insert(parent_node);
                if (solver.debug_logs()) { spdlog::debug("Updating state " + parent_node->state.print() +
                                                        ": value=" + StringConverter::from(parent_node->value) +
                                                        ", visits=" + StringConverter::from(parent_node->visits_count) +
                                                        Tsolver::ExecutionPolicy::print_thread()); }
            }, a->parent->mutex);
        }
    }
};


template <typename Tdomain,
          typename TexecutionPolicy = SequentialExecution,
          template <typename Tsolver> class TtransitionMode = DistributionTransitionMode,
          template <typename Tsolver> class TtreePolicy = DefaultTreePolicy,
          template <typename Tsolver> class Texpander = FullExpand,
          template <typename Tsolver> class TactionSelectorOptimization = UCB1ActionSelector,
          template <typename Tsolver> class TactionSelectorExecution = BestQValueActionSelector,
          template <typename Tsolver> class TrolloutPolicy = DefaultRolloutPolicy,
          template <typename Tsolver> class TbackPropagator = GraphBackup>
class MCTSSolver {
public :
    typedef MCTSSolver<Tdomain, TexecutionPolicy,
                       TtransitionMode, TtreePolicy, Texpander,
                       TactionSelectorOptimization, TactionSelectorExecution,
                       TrolloutPolicy, TbackPropagator> Solver;

    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef TexecutionPolicy ExecutionPolicy;
    typedef TtransitionMode<Solver> TransitionMode;
    typedef TtreePolicy<Solver> TreePolicy;
    typedef Texpander<Solver> Expander;
    typedef TactionSelectorOptimization<Solver> ActionSelectorOptimization;
    typedef TactionSelectorExecution<Solver> ActionSelectorExecution;
    typedef TrolloutPolicy<Solver> RolloutPolicy;
    typedef TbackPropagator<Solver> BackPropagator;

    typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
    typedef typename ExecutionPolicy::template atomic<double> atomic_double;
    typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

    struct StateNode;

    struct ActionNode {
        Action action;
        typedef std::unordered_map<StateNode*, std::pair<double, std::size_t>> OutcomeMap; // next state nodes owned by _graph
        OutcomeMap outcomes;
        std::vector<typename OutcomeMap::iterator> dist_to_outcome;
        std::discrete_distribution<> dist;
        atomic_size_t expansions_count; // used only for partial expansion mode
        atomic_double value;
        atomic_size_t visits_count;
        StateNode* parent;

        ActionNode(const Action& a)
            : action(a), expansions_count(0), value(0.0),
              visits_count(0), parent(nullptr) {}
        
        ActionNode(const ActionNode& a)
            : action(a.action), outcomes(a.outcomes), dist_to_outcome(a.dist_to_outcome),
              dist(a.dist), expansions_count((std::size_t) a.expansions_count),
              value((double) a.value), visits_count((std::size_t) a.visits_count),
              parent(a.parent) {}
        
        struct Key {
            const Action& operator()(const ActionNode& an) const { return an.action; }
        };
    };

    struct StateNode {
        typedef typename SetTypeDeducer<ActionNode, Action>::Set ActionSet;
        State state;
        atomic_bool terminal;
        atomic_bool expanded; // used only for full expansion mode
        atomic_size_t expansions_count; // used only for partial expansion mode
        ActionSet actions;
        atomic_double value;
        atomic_size_t visits_count;
        std::unordered_set<ActionNode*> parents;
        mutable typename ExecutionPolicy::Mutex mutex;

        StateNode(const State& s)
            : state(s), terminal(false), expanded(false),
              expansions_count(0), value(0.0), visits_count(0) {}
        
        StateNode(const StateNode& s)
            : state(s.state), terminal((bool) s.terminal), expanded((bool) s.expanded),
              expansions_count((std::size_t) s.expansions_count), actions(s.actions),
              value((double) s.value), visits_count((std::size_t) s.visits_count),
              parents(s.parents) {}
        
        struct Key {
            const State& operator()(const StateNode& sn) const { return sn.state; }
        };
    };

    typedef typename SetTypeDeducer<StateNode, State>::Set Graph;

    MCTSSolver(Domain& domain,
               std::size_t time_budget = 3600000,
               std::size_t rollout_budget = 100000,
               std::size_t max_depth = 1000,
               double discount = 1.0,
               bool online_node_garbage = false,
               bool debug_logs = false,
               std::unique_ptr<TreePolicy> tree_policy = std::make_unique<TreePolicy>(),
               std::unique_ptr<Expander> expander = std::make_unique<Expander>(),
               std::unique_ptr<ActionSelectorOptimization> action_selector_optimization = std::make_unique<ActionSelectorOptimization>(),
               std::unique_ptr<ActionSelectorExecution> action_selector_execution = std::make_unique<ActionSelectorExecution>(),
               std::unique_ptr<RolloutPolicy> rollout_policy = std::make_unique<RolloutPolicy>(),
               std::unique_ptr<BackPropagator> back_propagator = std::make_unique<BackPropagator>())
    : _domain(domain),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _discount(discount), _nb_rollouts(0),
      _online_node_garbage(online_node_garbage),
      _debug_logs(debug_logs),
      _tree_policy(std::move(tree_policy)),
      _expander(std::move(expander)),
      _action_selector_optimization(std::move(action_selector_optimization)),
      _action_selector_execution(std::move(action_selector_execution)),
      _rollout_policy(std::move(rollout_policy)),
      _back_propagator(std::move(back_propagator)),
      _current_state(nullptr) {
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
            spdlog::info("Running " + ExecutionPolicy::print_type() + " MCTS solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();
            _nb_rollouts = 0;

            // Get the root node
            auto si = _graph.emplace(s);
            StateNode& root_node = const_cast<StateNode&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe

            boost::integer_range<std::size_t> parallel_rollouts(0, _domain.get_parallel_capacity());

            std::for_each(ExecutionPolicy::policy, parallel_rollouts.begin(), parallel_rollouts.end(), [this, &start_time, &root_node] (const std::size_t& thread_id) {
                
                while (elapsed_time(start_time) < _time_budget && _nb_rollouts < _rollout_budget) {
                
                    std::size_t depth = 0;
                    StateNode* sn = (*_tree_policy)(*this, &thread_id, *_expander, *_action_selector_optimization, root_node, depth);
                    (*_rollout_policy)(*this, &thread_id, *sn, depth);
                    (*_back_propagator)(*this, &thread_id, *sn);
                    _nb_rollouts++;

                }
                
            });

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            spdlog::info("MCTS finished to solve from state " + s.print() +
                         " in " + StringConverter::from((double) duration / (double) 1e9) + " seconds with " +
                         StringConverter::from(_nb_rollouts) + " rollouts.");
        } catch (const std::exception& e) {
            spdlog::error("MCTS failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) {
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            return false;
        } else {
            return (*_action_selector_execution)(*this, nullptr, *si) != nullptr;
        }
    }

    Action get_best_action(const State& s) {
        auto si = _graph.find(s);
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = (*_action_selector_execution)(*this, nullptr, *si);
        }
        if (action == nullptr) {
            spdlog::error("SKDECIDE exception: no best action found in state " + s.print());
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        } else {
            if (_debug_logs) {
                std::string str = "(";
                for (const auto& o : action->outcomes) {
                    str += "\n    " + o.first->state.print();
                }
                str += "\n)";
                spdlog::debug("Best action's known outcomes:\n" + str);
            }
            if (_online_node_garbage && _current_state) {
                std::unordered_set<StateNode*> root_subgraph, child_subgraph;
                compute_reachable_subgraph(_current_state, root_subgraph);
                compute_reachable_subgraph(const_cast<StateNode*>(&(*si)), child_subgraph); // we won't change the real key (StateNode::state) so we are safe
                remove_subgraph(root_subgraph, child_subgraph);
            }
            _current_state = const_cast<StateNode*>(&(*si)); // we won't change the real key (StateNode::state) so we are safe
            _action_prefix.push_back(action->action);
            return action->action;
        }
    }

    double get_best_value(const State& s) {
        auto si = _graph.find(StateNode(s));
        ActionNode* action = nullptr;
        if (si != _graph.end()) {
            action = (*_action_selector_execution)(*this, nullptr, *si);
        }
        if (action == nullptr) {
            spdlog::error("SKDECIDE exception: no best action found in state " + s.print());
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
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
            ActionNode* action = (*_action_selector_execution)(*this, nullptr, n);
            if (action != nullptr) {
                p.insert(std::make_pair(n.state, std::make_pair(action->action, (double) action->value)));
            }
        }
        return p;
    }

    Domain& domain() { return _domain; }

    std::size_t time_budget() const { return _time_budget; }

    std::size_t rollout_budget() const { return _rollout_budget; }

    std::size_t max_depth() const { return _max_depth; }

    double discount() const { return _discount; }

    ExecutionPolicy& execution_policy() { return _execution_policy; }

    TransitionMode& transition_mode() { return _transition_mode; }

    const TreePolicy& tree_policy() { return *_tree_policy; }

    const Expander& expander() { return *_expander; }

    const ActionSelectorOptimization& action_selector_optimization() { return *_action_selector_optimization; }

    const ActionSelectorExecution& action_selector_execution() { return *_action_selector_execution; }

    const RolloutPolicy& rollout_policy() { return *_rollout_policy; }

    const BackPropagator& back_propagator() { return *_back_propagator; }

    Graph& graph() { return _graph; }

    const std::list<Action>& action_prefix() const { return _action_prefix; }

    std::mt19937& gen() { return *_gen; }

    typename ExecutionPolicy::Mutex& gen_mutex() { return _gen_mutex; }

    bool debug_logs() const { return _debug_logs; }

private :

    Domain& _domain;
    atomic_size_t _time_budget;
    atomic_size_t _rollout_budget;
    atomic_size_t _max_depth;
    atomic_double _discount;
    atomic_size_t _nb_rollouts;
    bool _online_node_garbage;
    atomic_bool _debug_logs;

    ExecutionPolicy _execution_policy;
    TransitionMode _transition_mode;

    std::unique_ptr<TreePolicy> _tree_policy;
    std::unique_ptr<Expander> _expander;
    std::unique_ptr<ActionSelectorOptimization> _action_selector_optimization;
    std::unique_ptr<ActionSelectorExecution> _action_selector_execution;
    std::unique_ptr<RolloutPolicy> _rollout_policy;
    std::unique_ptr<BackPropagator> _back_propagator;

    Graph _graph;
    StateNode* _current_state;
    std::list<Action> _action_prefix;

    std::unique_ptr<std::mt19937> _gen;
    typename ExecutionPolicy::Mutex _gen_mutex;
    typename ExecutionPolicy::Mutex _time_mutex;

    void compute_reachable_subgraph(StateNode* node, std::unordered_set<StateNode*>& subgraph) {
        std::unordered_set<StateNode*> frontier;
        frontier.insert(node);
        subgraph.insert(node);
        while(!frontier.empty()) {
            std::unordered_set<StateNode*> new_frontier;
            for (auto& n : frontier) {
                for (auto& action : n->actions) {
                    for (auto& outcome : action.outcomes) {
                        if (subgraph.find(outcome.first) == subgraph.end()) {
                            new_frontier.insert(outcome.first);
                            subgraph.insert(outcome.first);
                        }
                    }
                }
            }
            frontier = new_frontier;
        }
    }

    void remove_subgraph(std::unordered_set<StateNode*>& root_subgraph, std::unordered_set<StateNode*>& child_subgraph) {
        std::unordered_set<StateNode*> removed_subgraph;
        // First pass: look for nodes in root_subgraph but not child_subgraph and remove
        // those nodes from their children's parents
        // Don't actually remove those nodes in the first pass otherwise some children to remove
        // won't exist anymore when looking for their parents
        for (auto& n : root_subgraph) {
            if (child_subgraph.find(n) == child_subgraph.end()) {
                for (auto& action : n->actions) {
                    for (auto& outcome : action.outcomes) {
                        // we won't change the real key (ActionNode::action) so we are safe
                        outcome.first->parents.erase(&const_cast<ActionNode&>(action));
                    }
                }
                removed_subgraph.insert(n);
            }
        }
        // Second pass: actually remove nodes in root_subgraph but not in child_subgraph
        for (auto& n : removed_subgraph) {
            _graph.erase(StateNode(n->state));
        }
    }

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
}; // MCTSSolver class

/** UCT is MCTS with the default template options */
template <typename Tdomain,
          typename Texecution_policy,
          template <typename Tsolver> typename TtransitionMode,
          template <typename Tsolver> typename ...T>
using UCTSolver = MCTSSolver<Tdomain, Texecution_policy, TtransitionMode, T...>;

} // namespace skdecide

#endif // SKDECIDE_MCTS_HH
