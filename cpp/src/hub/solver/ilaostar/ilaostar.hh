/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the skdecide implementation of ILAO* from the paper
 * "LAO*: A heuristic search algorithm that finds solutions with loops"
 * from Hansen and Zilberstein (AIJ 2001). This is actually the improved
 * version of LAO* as described in Table 7 of the aforementioned paper.
 */
#ifndef SKDECIDE_ILAOSTAR_HH
#define SKDECIDE_ILAOSTAR_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <stack>
#include <list>
#include <chrono>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class ILAOStarSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  ILAOStarSolver(
      Domain &domain,
      const std::function<Predicate(Domain &, const State &)> &goal_checker,
      const std::function<Value(Domain &, const State &)> &heuristic,
      double discount = 1.0, double epsilon = 0.001, bool debug_logs = false);

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s using heuristic function h
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  double get_best_value(const State &s) const;
  std::size_t get_nb_of_explored_states() const;
  std::size_t best_solution_graph_size() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const;

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  std::function<bool(Domain &, const State &)> _goal_checker;
  std::function<Value(Domain &, const State &)> _heuristic;
  atomic_double _discount;
  atomic_double _epsilon;
  bool _debug_logs;
  ExecutionPolicy _execution_policy;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    atomic_double best_value;
    atomic_double first_passage_time;
    double residual;
    bool goal;
    atomic_bool reach_tip_node;
    bool solved;

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

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::unordered_set<StateNode *> _best_solution_graph;

  void expand(StateNode &s);
  void depth_first_search(StateNode &s);
  void compute_best_solution_graph(StateNode &s);
  double update(StateNode &s);
  void value_iteration();
  bool update_reachability(StateNode &s);
  void compute_reachability();
  double update_mfpt(StateNode &s);
  void compute_mfpt();
  void update_solved_bits();
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/ilaostar_impl.hh"
#endif

#endif // SKDECIDE_ILAOSTAR_HH
