/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_GPCI_HH
#define SKDECIDE_GPCI_HH

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class GPCISolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  enum class Phase { ENUMERATION, PROBABILITY, COST };

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<bool(const GPCISolver &, Domain &)> CallbackFunctor;

  GPCISolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      double epsilon = 0.001,
      const CallbackFunctor &callback = [](const GPCISolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  double get_goal_probability(const State &s) const;
  double get_goal_cost(const State &s) const;

  Phase get_current_phase() const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_prob_iterations() const;
  std::size_t get_nb_cost_iterations() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const;

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;

  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  atomic_double _epsilon;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    atomic_double goal_probability;
    atomic_double goal_cost;
    bool terminal;
    bool goal;

    StateNode(const State &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>> outcomes;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::vector<StateNode *> _non_goal_states;
  Phase _current_phase;
  std::size_t _nb_prob_iterations;
  std::size_t _nb_cost_iterations;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void enumerate_reachable_states(const State &s);
  void expand(StateNode &s);
  double probability_update(StateNode &s);
  double cost_update(StateNode &s);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/gpci_impl.hh"
#endif

#endif // SKDECIDE_GPCI_HH
