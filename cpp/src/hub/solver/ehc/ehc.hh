/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_EHC_HH
#define SKDECIDE_EHC_HH

#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class EHCSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<std::vector<Action>(Domain &, const State &)>
      PreferredActionsFunctor;
  typedef std::function<bool(const EHCSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Constructs a new Enforced Hill Climbing (EHC) solver.
   *
   * EHC performs breadth-first search from the current state for a successor
   * with strictly lower heuristic value, commits the path, and repeats until
   * a goal is reached. When preferred actions are provided, they are expanded
   * first in each BFS layer.
   *
   * @param domain The domain instance to solve.
   * @param goal_checker Functor testing whether a state is a goal.
   * @param heuristic Functor returning the heuristic cost estimate for a state.
   * @param preferred_actions Optional functor returning a list of preferred
   *   actions to expand first in BFS. Defaults to nullptr (no preference).
   * @param callback Functor called after each EHC improvement step, taking the
   *   solver and domain as arguments and returning true to stop the search.
   *   Defaults to always returning false.
   * @param verbose Whether to log verbose debug messages. Defaults to false.
   */
  EHCSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic,
      const PreferredActionsFunctor &preferred_actions = nullptr,
      const CallbackFunctor &callback = [](const EHCSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;
  std::size_t get_nb_explored_states() const;
  typename SetTypeDeducer<State>::Set get_explored_states() const;
  std::size_t get_solving_time() const;

  std::vector<std::tuple<State, Action, Value>>
  get_plan(const State &from_state) const;

  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map
  get_policy() const;

  template <typename Params>
  static std::unique_ptr<EHCSolver> create_from_params(
      Domain &domain,
      std::function<Predicate(Domain &, const State &)> goal_checker,
      std::function<Value(Domain &, const State &)> heuristic,
      std::function<Value(const State &)> terminal_value, const Params &params,
      bool verbose);

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  PreferredActionsFunctor _preferred_actions;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  struct Node {
    State state;
    std::tuple<Node *, Action, double> best_parent;
    std::pair<Action *, Node *> best_action;
    bool solved = false;

    Node(const State &s);

    struct Key {
      const State &operator()(const Node &n) const;
    };
  };

  typedef typename SetTypeDeducer<Node, State>::Set Graph;
  Graph _graph;

  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/ehc_impl.hh"
#endif

#endif // SKDECIDE_EHC_HH
