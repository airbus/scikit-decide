/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_AOSTAR_HH
#define SKDECIDE_AOSTAR_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <list>
#include <queue>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

/**
 * @brief This is the skdecide implementation of the AO* algorithm for searching
 * cost-minimal policies in additive AND/OR graphs with admissible heuristics
 * as described in "Principles of Artificial Intelligence" by Nilsson, N. (1980)
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (one of
 * 'SequentialExecution' to generate state-action transitions in sequence,
 * or 'ParallelExecution' to generate state-actio
 * transitions in parallel on different threads)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class AOStarSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const AOStarSolver &, Domain &)> CallbackFunctor;

  AOStarSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, double discount = 1.0,
      std::size_t max_tip_expansions = 1, bool detect_cycles = false,
      bool debug_logs = false,
      const CallbackFunctor &callback = [](const AOStarSolver &, Domain &) {
        return false;
      });

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s using heuristic function h
  void solve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;
  std::size_t get_nb_explored_states() const;
  typename SetTypeDeducer<State>::Set get_explored_states() const;
  std::size_t get_nb_tip_states() const;
  const State &get_top_tip_state() const;
  std::size_t get_solving_time() const;
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map
  get_policy() const;

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  double _discount;
  std::size_t _max_tip_expansions;
  bool _detect_cycles;
  bool _debug_logs;
  CallbackFunctor _callback;
  ExecutionPolicy _execution_policy;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    double best_value;
    bool solved;
    std::list<ActionNode *> parents;

    StateNode(const State &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>>
        outcomes; // next state nodes owned by _graph
    double value;
    StateNode *parent;

    ActionNode(const Action &a);
  };

  struct StateNodeCompare {
    bool operator()(StateNode *&a, StateNode *&b) const;
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;

  typedef std::priority_queue<StateNode *, std::vector<StateNode *>,
                              StateNodeCompare>
      PriorityQueue;
  PriorityQueue _priority_queue;

  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/aostar_impl.hh"
#endif

#endif // SKDECIDE_AOSTAR_HH
