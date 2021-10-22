/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_FULL_EXPAND_IMPL_HH
#define SKDECIDE_MCTS_FULL_EXPAND_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === FullExpand implementation ===

template <typename Tsolver>
struct FullExpand<Tsolver>::ExpandActionImplementation {
  template <typename Ttransition_mode = typename Tsolver::TransitionMode,
            typename Enable = void>
  struct Impl {};
};

template <typename Tsolver>
template <typename Ttransition_mode>
struct FullExpand<Tsolver>::ExpandActionImplementation::Impl<
    Ttransition_mode,
    typename std::enable_if<std::is_same<
        Ttransition_mode, DistributionTransitionMode<Tsolver>>::value>::type>
    : public ExpandActionImplementation {

  const HeuristicFunctor &_heuristic;

  Impl(const HeuristicFunctor &heuristic) : _heuristic(heuristic) {}

  virtual ~Impl() {}

  typename Tsolver::StateNode *
  expand_action(Tsolver &solver, const std::size_t *thread_id,
                typename Tsolver::StateNode &state,
                typename Tsolver::ActionNode &action) const {
    try {
      // Generate the next states of this action
      auto next_states = solver.domain()
                             .get_next_state_distribution(
                                 state.state, action.action, thread_id)
                             .get_values();
      std::vector<typename Tsolver::StateNode *> untried_outcomes;
      std::vector<double> weights;
      std::vector<double> outcome_weights;

      for (auto ns : next_states) {
        std::pair<typename Tsolver::Graph::iterator, bool> i;

        solver.execution_policy().protect(
            [&i, &solver, &ns]() { i = solver.graph().emplace(ns.state()); });

        typename Tsolver::StateNode &next_node =
            const_cast<typename Tsolver::StateNode &>(
                *(i.first)); // we won't change the real key (StateNode::state)
                             // so we are safe
        double reward = 0.0;

        solver.execution_policy().protect(
            [&reward, &solver, &state, &action, &next_node, &thread_id]() {
              reward = solver.domain()
                           .get_transition_value(state.state, action.action,
                                                 next_node.state, thread_id)
                           .reward();
            },
            next_node.mutex);

        solver.execution_policy().protect(
            [&action, &next_node, &outcome_weights, &reward, &ns]() {
              auto ii = action.outcomes.insert(
                  std::make_pair(&next_node, std::make_pair(reward, 1)));

              if (ii.second) { // new outcome
                action.dist_to_outcome.push_back(ii.first);
                outcome_weights.push_back(ns.probability());
              } else { // existing outcome (following code not efficient but
                       // hopefully very rare case if domain is well defined)
                for (unsigned int oid = 0; oid < outcome_weights.size();
                     oid++) {
                  if (action.dist_to_outcome[oid]->first ==
                      ii.first->first) { // found my outcome!
                    std::pair<double, std::size_t> &mp = ii.first->second;
                    mp.first =
                        ((double)(outcome_weights[oid] * mp.first) +
                         (reward * ns.probability())) /
                        ((double)(outcome_weights[oid] + ns.probability()));
                    outcome_weights[oid] += ns.probability();
                    mp.second += 1; // useless in this mode a priori, but just
                                    // keep track for coherency
                    break;
                  }
                }
              }
            },
            action.parent->mutex);

        solver.execution_policy().protect(
            [this, &next_node, &action, &i, &solver, &thread_id,
             &untried_outcomes, &weights, &ns]() {
              next_node.parents.insert(&action);

              if (i.second) { // new node
                next_node.terminal =
                    solver.domain().is_terminal(next_node.state, thread_id);
                std::pair<typename Tsolver::Domain::Value, std::size_t> h =
                    _heuristic(solver.domain(), next_node.state, thread_id);
                next_node.value = h.first.reward();
                next_node.visits_count = h.second;
              }

              if (next_node.actions.empty()) {
                if (solver.debug_logs())
                  Logger::debug(
                      "Candidate next state: " + next_node.state.print() +
                      Tsolver::ExecutionPolicy::print_thread());
                untried_outcomes.push_back(&next_node);
                weights.push_back(ns.probability());
              }
            },
            next_node.mutex);
      }

      // Record the action's outcomes distribution
      solver.execution_policy().protect(
          [&action, &outcome_weights]() {
            action.dist = std::discrete_distribution<>(outcome_weights.begin(),
                                                       outcome_weights.end());
          },
          action.parent->mutex);

      // Pick a random next state
      if (untried_outcomes.empty()) {
        // All next states already visited => pick a random next state using
        // action.dist
        std::size_t outcome_id = 0;

        solver.execution_policy().protect(
            [&outcome_id, &action, &solver]() {
              outcome_id = action.dist(solver.gen());
            },
            solver.gen_mutex());

        typename Tsolver::StateNode *outcome = nullptr;

        solver.execution_policy().protect(
            [&action, &outcome, &outcome_id]() {
              outcome = action.dist_to_outcome[outcome_id]->first;
            },
            action.parent->mutex);

        return outcome;
      } else {
        // Pick a random next state among untried ones
        std::discrete_distribution<> odist(weights.begin(), weights.end());
        std::size_t outcome_id = 0;

        solver.execution_policy().protect(
            [&outcome_id, &odist, &solver]() {
              outcome_id = odist(solver.gen());
            },
            solver.gen_mutex());

        return untried_outcomes[outcome_id];
      }
    } catch (const std::exception &e) {
      solver.execution_policy().protect(
          [&action, &e]() {
            Logger::error("SKDECIDE exception in MCTS when expanding action " +
                          action.action.print() + ": " + e.what() +
                          Tsolver::ExecutionPolicy::print_thread());
          },
          action.parent->mutex);
      throw;
    }
  }
};

template <typename Tsolver>
template <typename Ttransition_mode>
struct FullExpand<Tsolver>::ExpandActionImplementation::Impl<
    Ttransition_mode,
    typename std::enable_if<
        std::is_same<Ttransition_mode, StepTransitionMode<Tsolver>>::value ||
        std::is_same<Ttransition_mode, SampleTransitionMode<Tsolver>>::value>::
        type> : public ExpandActionImplementation {

  const HeuristicFunctor &_heuristic;
  mutable typename Tsolver::ExecutionPolicy::template atomic<bool>
      _checked_transition_mode;

  Impl(const HeuristicFunctor &heuristic)
      : _heuristic(heuristic), _checked_transition_mode(false) {}

  virtual ~Impl() {}

  typename Tsolver::StateNode *
  expand_action(Tsolver &solver, const std::size_t *thread_id,
                typename Tsolver::StateNode &state,
                typename Tsolver::ActionNode &action) const {
    try {
      if (!_checked_transition_mode) {
        Logger::warn("Using MCTS full expansion mode with step() or sample() "
                     "domain's transition mode assumes the domain is "
                     "deterministic (unpredictable result otherwise).");
        _checked_transition_mode = true;
      }
      // Generate the next state of this action
      typename Tsolver::Domain::EnvironmentOutcome to =
          solver.transition_mode().random_next_outcome(
              solver, thread_id, state.state, action.action);
      std::pair<typename Tsolver::Graph::iterator, bool> i;

      solver.execution_policy().protect([&i, &solver, &to]() {
        i = solver.graph().emplace(to.observation());
      });

      typename Tsolver::StateNode &next_node =
          const_cast<typename Tsolver::StateNode &>(
              *(i.first)); // we won't change the real key (StateNode::state) so
                           // we are safe

      solver.execution_policy().protect(
          [&action, &next_node, &to]() {
            auto ii = action.outcomes.insert(std::make_pair(
                &next_node, std::make_pair(to.transition_value().reward(), 1)));
            action.dist_to_outcome.push_back(ii.first);
          },
          action.parent->mutex);

      solver.execution_policy().protect(
          [this, &next_node, &action, &i, &to, &solver, &thread_id]() {
            next_node.parents.insert(&action);

            if (i.second) { // new node
              next_node.terminal = to.termination();
              std::pair<typename Tsolver::Domain::Value, std::size_t> h =
                  _heuristic(solver.domain(), next_node.state, thread_id);
              next_node.value = h.first.reward();
              next_node.visits_count = h.second;
            }
          },
          next_node.mutex);

      // Record the action's outcomes distribution
      solver.execution_policy().protect(
          [&action]() { action.dist = std::discrete_distribution<>({1.0}); },
          action.parent->mutex);

      if (solver.debug_logs()) {
        solver.execution_policy().protect(
            [&next_node]() {
              Logger::debug("Candidate next state: " + next_node.state.print() +
                            Tsolver::ExecutionPolicy::print_thread());
            },
            next_node.mutex);
      }

      return &next_node;
    } catch (const std::exception &e) {
      solver.execution_policy().protect(
          [&action, &e]() {
            Logger::error("SKDECIDE exception in MCTS when expanding action " +
                          action.action.print() + ": " + e.what() +
                          Tsolver::ExecutionPolicy::print_thread());
          },
          action.parent->mutex);
      throw;
    }
  }
};

#define SK_MCTS_FULL_EXPAND_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_FULL_EXPAND_CLASS FullExpand<Tsolver>

SK_MCTS_FULL_EXPAND_TEMPLATE_DECL
SK_MCTS_FULL_EXPAND_CLASS::FullExpand(const HeuristicFunctor &heuristic)
    : _heuristic(heuristic) {
  _action_expander =
      std::make_unique<typename ExpandActionImplementation::template Impl<>>(
          _heuristic);
}

SK_MCTS_FULL_EXPAND_TEMPLATE_DECL
SK_MCTS_FULL_EXPAND_CLASS::FullExpand(const FullExpand &other)
    : _heuristic(other._heuristic) {
  _action_expander =
      std::make_unique<typename ExpandActionImplementation::template Impl<>>(
          _heuristic);
}

SK_MCTS_FULL_EXPAND_TEMPLATE_DECL
SK_MCTS_FULL_EXPAND_CLASS::~FullExpand() = default;

SK_MCTS_FULL_EXPAND_TEMPLATE_DECL
typename Tsolver::StateNode *
SK_MCTS_FULL_EXPAND_CLASS::operator()(Tsolver &solver,
                                      const std::size_t *thread_id,
                                      typename Tsolver::StateNode &n) const {
  try {
    if (solver.debug_logs()) {
      solver.execution_policy().protect(
          [&n]() {
            Logger::debug("Testing expansion of state " + n.state.print() +
                          Tsolver::ExecutionPolicy::print_thread());
          },
          n.mutex);
    }

    if (n.expanded) {
      if (solver.debug_logs()) {
        Logger::debug("State already fully expanded" +
                      Tsolver::ExecutionPolicy::print_thread());
      }
      return nullptr;
    }

    // Generate applicable actions if not already done
    solver.execution_policy().protect(
        [&n, &solver, &thread_id]() {
          if (n.actions.empty()) {
            if (solver.debug_logs()) {
              Logger::debug(
                  "State never expanded, generating all next actions" +
                  Tsolver::ExecutionPolicy::print_thread());
            }
            auto applicable_actions =
                solver.domain()
                    .get_applicable_actions(n.state, thread_id)
                    .get_elements();

            for (const auto &a : applicable_actions) {
              auto i = n.actions.emplace(typename Tsolver::ActionNode(a));

              if (i.second) {
                // we won't change the real key (ActionNode::action) so we are
                // safe
                const_cast<typename Tsolver::ActionNode &>(*i.first).parent =
                    &n;
              }
            }
          }
        },
        n.mutex);

    // Check for untried outcomes
    if (solver.debug_logs()) {
      Logger::debug("Checking for untried outcomes" +
                    Tsolver::ExecutionPolicy::print_thread());
    }
    std::vector<std::pair<typename Tsolver::ActionNode *,
                          typename Tsolver::StateNode *>>
        untried_outcomes;
    std::vector<double> weights;

    solver.execution_policy().protect(
        [&n, &untried_outcomes, &weights]() {
          for (auto &a : n.actions) {
            // we won't change the real key (ActionNode::action) so we are safe
            typename Tsolver::ActionNode &ca =
                const_cast<typename Tsolver::ActionNode &>(a);

            if (a.outcomes.empty()) {
              // we won't change the real key (ActionNode::action) so we are
              // safe
              untried_outcomes.push_back(std::make_pair(&ca, nullptr));
              weights.push_back(1.0);
            } else {
              // Check if there are next states that have been never visited
              std::vector<double> probs = a.dist.probabilities();

              for (std::size_t p = 0; p < probs.size(); p++) {
                typename Tsolver::StateNode *on = ca.dist_to_outcome[p]->first;

                if (on->visits_count == 0) {
                  untried_outcomes.push_back(std::make_pair(&ca, on));
                  weights.push_back(probs[p]);
                }
              }
            }
          }
        },
        n.mutex);

    if (untried_outcomes.empty()) { // nothing to expand
      if (solver.debug_logs()) {
        Logger::debug("All outcomes already tried" +
                      Tsolver::ExecutionPolicy::print_thread());
      }
      n.expanded = true;
      return nullptr;
    } else {
      std::discrete_distribution<> odist(weights.begin(), weights.end());
      std::size_t outcome_id = 0;
      solver.execution_policy().protect(
          [&outcome_id, &odist, &solver]() {
            outcome_id = odist(solver.gen());
          },
          solver.gen_mutex());
      auto &uo = untried_outcomes[outcome_id];

      if (uo.second == nullptr) { // unexpanded action
        if (solver.debug_logs()) {
          Logger::debug(
              "Found one unexpanded action: " + uo.first->action.print() +
              Tsolver::ExecutionPolicy::print_thread());
        }
        return expand_action(solver, thread_id, n, *(uo.first));
      } else { // expanded action, just return the selected next state
        if (solver.debug_logs()) {
          solver.execution_policy().protect(
              [&uo]() {
                Logger::debug("Found one untried outcome: action " +
                              uo.first->action.print() + " and next state " +
                              uo.second->state.print() +
                              Tsolver::ExecutionPolicy::print_thread());
              },
              uo.second->mutex);
        }
        return uo.second;
      }
    }
  } catch (const std::exception &e) {
    solver.execution_policy().protect(
        [&n, &e]() {
          Logger::error("SKDECIDE exception in MCTS when expanding state " +
                        n.state.print() + ": " + e.what() +
                        Tsolver::ExecutionPolicy::print_thread());
        },
        n.mutex);
    throw;
  }
}

SK_MCTS_FULL_EXPAND_TEMPLATE_DECL
typename Tsolver::StateNode *SK_MCTS_FULL_EXPAND_CLASS::expand_action(
    Tsolver &solver, const std::size_t *thread_id,
    typename Tsolver::StateNode &state,
    typename Tsolver::ActionNode &action) const {
  return static_cast<typename ExpandActionImplementation::template Impl<> &>(
             *_action_expander)
      .expand_action(solver, thread_id, state, action);
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_FULL_EXPAND_IMPL_HH
