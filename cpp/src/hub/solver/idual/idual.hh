/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_IDUAL_HH
#define SKDECIDE_IDUAL_HH

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Highs.h"
#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

template <typename T, typename = void>
struct has_get_constraints : std::false_type {};

template <typename T>
struct has_get_constraints<
    T, std::void_t<decltype(std::declval<T>().get_constraints())>>
    : std::true_type {};

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class IDualSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const IDualSolver &, Domain &)> CallbackFunctor;
  typedef std::function<double(Domain &, const State &, std::size_t)>
      SecondaryHeuristicFunctor;

  IDualSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(1000.0, false); },
      const SecondaryHeuristicFunctor &secondary_heuristic = nullptr,
      const std::vector<double> &dead_end_costs = {}, double epsilon = 0.001,
      double lp_infinity = 1e20, double default_dead_end_cost = 1000.0,
      const CallbackFunctor &callback = [](const IDualSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  Value get_best_value(const State &s) const;

  template <typename D = Domain,
            std::enable_if_t<!has_get_constraints<D>::value, int> = 0>
  const Action &get_best_action(const State &s) const;

  template <typename D = Domain,
            std::enable_if_t<has_get_constraints<D>::value, int> = 0>
  std::vector<std::pair<Action, double>>
  get_action_distribution(const State &s) const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_lp_iterations() const;
  std::size_t get_solving_time() const;
  typename SetTypeDeducer<State>::Set get_explored_states() const;

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  SecondaryHeuristicFunctor _secondary_heuristic;
  double _epsilon;
  double _lp_infinity;
  double _default_dead_end_cost;
  CallbackFunctor _callback;
  bool _verbose;

  std::size_t _n_constraints;
  std::vector<double> _cost_bounds;
  std::vector<double> _dead_end_costs;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    double best_value;
    bool expanded;
    bool goal;
    bool terminal;
    std::size_t index;
    std::vector<std::pair<ActionNode *, double>> action_probabilities;

    struct Key {
      const State &operator()(const StateNode &sn) const { return sn.state; }
    };

    StateNode(const State &s);
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>> outcomes;
    std::vector<double> secondary_costs;
    double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;

  ExecutionPolicy _execution_policy;

  std::vector<StateNode *> _fringe_reachable;

  std::size_t _nb_lp_iterations;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  // --- Incremental LP state ---
  std::unique_ptr<Highs> _highs;

  struct LPColInfo {
    StateNode *sn;
    ActionNode *an;
  };
  std::vector<LPColInfo> _lp_col_info;
  std::unordered_map<StateNode *, std::vector<HighsInt>> _lp_state_sa_cols;
  std::unordered_map<StateNode *, HighsInt> _lp_state_xd_col;

  std::unordered_map<StateNode *, HighsInt> _lp_flow_row;
  HighsInt _lp_c9_row = -1;
  std::vector<HighsInt> _lp_c11_rows;

  struct SuccEntry {
    HighsInt col;
    double prob;
  };
  std::unordered_map<StateNode *, std::vector<SuccEntry>> _lp_succ_to_cols;

  std::vector<double> _lp_col_obj;
  std::vector<double> _lp_col_c9;
  std::vector<std::vector<double>> _lp_col_c11;

  bool _lp_initialized = false;

  void expand_states(std::vector<StateNode *> &fr);
  void init_lp(const State &s0);
  void update_lp(const State &s0,
                 const std::vector<StateNode *> &newly_expanded);
  void extract_solution();

  double compute_sa_obj_cost(ActionNode *an) const;
  double compute_sa_c9_coeff(ActionNode *an) const;
  void add_sa_column(StateNode *sn, ActionNode *an, const State &s0);
  void add_xd_column(StateNode *sn);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/idual_impl.hh"
#endif

#endif // SKDECIDE_IDUAL_HH
