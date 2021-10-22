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

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class AOStarSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  AOStarSolver(
      Domain &domain,
      const std::function<Predicate(Domain &, const State &)> &goal_checker,
      const std::function<Value(Domain &, const State &)> &heuristic,
      double discount = 1.0, std::size_t max_tip_expansions = 1,
      bool detect_cycles = false, bool debug_logs = false);

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s using heuristic function h
  void solve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  const double &get_best_value(const State &s) const;

private:
  Domain &_domain;
  std::function<bool(Domain &, const State &)> _goal_checker;
  std::function<Value(Domain &, const State &)> _heuristic;
  double _discount;
  std::size_t _max_tip_expansions;
  bool _detect_cycles;
  bool _debug_logs;
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
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/aostar_impl.hh"
#endif

#endif // SKDECIDE_AOSTAR_HH
