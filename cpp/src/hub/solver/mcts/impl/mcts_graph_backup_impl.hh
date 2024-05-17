/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_GRAPH_BACKUP_IMPL_HH
#define SKDECIDE_MCTS_GRAPH_BACKUP_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === GraphBackup implementation ===

template <typename Tsolver>
struct GraphBackup<Tsolver>::UpdateFrontierImplementation {
  template <typename Texecution_policy = typename Tsolver::ExecutionPolicy,
            typename Enable = void>
  struct Impl {};

  template <typename Texecution_policy>
  struct Impl<Texecution_policy,
              typename std::enable_if<std::is_same<
                  Texecution_policy, SequentialExecution>::value>::type> {

    static void update_frontier(
        Tsolver &solver,
        std::unordered_set<typename Tsolver::StateNode *> &new_frontier,
        typename Tsolver::StateNode *f) {
      for (auto &a : f->parents) {
        double q_value =
            a->outcomes[f].first + (solver.discount() * (f->value));
        a->value = (((a->visits_count) * (a->value)) + q_value) /
                   ((double)(a->visits_count + 1));
        a->visits_count += 1;
        typename Tsolver::StateNode *parent_node = a->parent;
        parent_node->value =
            (((parent_node->visits_count) * (parent_node->value)) +
             (a->value)) /
            ((double)(parent_node->visits_count + 1));
        parent_node->visits_count += 1;
        new_frontier.insert(parent_node);
        if (solver.verbose()) {
          Logger::debug(
              "Updating state " + parent_node->state.print() +
              ": value=" + StringConverter::from(parent_node->value) +
              ", visits=" + StringConverter::from(parent_node->visits_count) +
              Tsolver::ExecutionPolicy::print_thread());
        }
      }
    }
  };

  template <typename Texecution_policy>
  struct Impl<Texecution_policy,
              typename std::enable_if<std::is_same<
                  Texecution_policy, ParallelExecution>::value>::type> {

    static void update_frontier(
        Tsolver &solver,
        std::unordered_set<typename Tsolver::StateNode *> &new_frontier,
        typename Tsolver::StateNode *f) {
      std::list<typename Tsolver::ActionNode *> parents;
      solver.execution_policy().protect(
          [&f, &parents]() {
            std::copy(f->parents.begin(), f->parents.end(),
                      std::inserter(parents, parents.end()));
          },
          f->mutex);
      for (auto &a : parents) {
        solver.execution_policy().protect(
            [&a, &solver, &f, &new_frontier]() {
              double q_value =
                  a->outcomes[f].first + (solver.discount() * (f->value));
              a->value = (((a->visits_count) * (a->value)) + q_value) /
                         ((double)(a->visits_count + 1));
              a->visits_count += 1;
              typename Tsolver::StateNode *parent_node = a->parent;
              parent_node->value =
                  (((parent_node->visits_count) * (parent_node->value)) +
                   (a->value)) /
                  ((double)(parent_node->visits_count + 1));
              parent_node->visits_count += 1;
              new_frontier.insert(parent_node);
              if (solver.verbose()) {
                Logger::debug(
                    "Updating state " + parent_node->state.print() +
                    ": value=" + StringConverter::from(parent_node->value) +
                    ", visits=" +
                    StringConverter::from(parent_node->visits_count) +
                    Tsolver::ExecutionPolicy::print_thread());
              }
            },
            a->parent->mutex);
      }
    }
  };
};

#define SK_MCTS_GRAPH_BACKUP_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_GRAPH_BACKUP_CLASS GraphBackup<Tsolver>

SK_MCTS_GRAPH_BACKUP_TEMPLATE_DECL
void SK_MCTS_GRAPH_BACKUP_CLASS::operator()(
    Tsolver &solver, const std::size_t *thread_id,
    typename Tsolver::StateNode &n) const {
  if (solver.verbose()) {
    solver.execution_policy().protect(
        [&n]() {
          Logger::debug("Back-propagating values from state " +
                        n.state.print() +
                        Tsolver::ExecutionPolicy::print_thread());
        },
        n.mutex);
  }

  std::size_t depth = 0; // used to prevent infinite loop in case of cycles
  std::unordered_set<typename Tsolver::StateNode *> frontier;
  frontier.insert(&n);

  while (!frontier.empty() && depth <= solver.max_depth()) {
    depth++;
    std::unordered_set<typename Tsolver::StateNode *> new_frontier;

    for (auto &f : frontier) {
      update_frontier(solver, new_frontier, f);
    }

    frontier = new_frontier;
  }
}

SK_MCTS_GRAPH_BACKUP_TEMPLATE_DECL
void SK_MCTS_GRAPH_BACKUP_CLASS::update_frontier(
    Tsolver &solver,
    std::unordered_set<typename Tsolver::StateNode *> &new_frontier,
    typename Tsolver::StateNode *f) {
  UpdateFrontierImplementation::template Impl<>::update_frontier(
      solver, new_frontier, f);
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_GRAPH_BACKUP_IMPL_HH
