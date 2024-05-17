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

/**
 * @brief This is the skdecide implementation of Improved-LAO* as described in
 * "LAO*: A heuristic search algorithm that finds solutions with loops" by Eric
 * A. Hansen and Shlomo Zilberstein (2001)
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (one of
 * 'SequentialExecution' to generate state-action transitions and to update
 * state attributes (e.g. Bellman residuals) in sequence, or 'ParallelExecution'
 * to generate state-action transitions and to update state attributes in
 * parallel on different threads)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class ILAOStarSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const ILAOStarSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new ILAOStarSolver object
   *
   * @param domain The domain instance
   * @param goal_checker Functor taking as arguments the domain and a state
   * object, and returning true if the state is the goal
   * @param heuristic Functor taking as arguments the domain and a state object,
   * and returning the heuristic estimate from the state to the goal
   * @param discount Value function's discount factor
   * @param epsilon Maximum Bellman error (residual) allowed to decide that a
   * state is solved
   * @param callback Functor called at the beginning of each policy update
   * depth-first search, taking as arguments the solver and the domain, and
   * returning true if the solver must be stopped
   * @param verbose Boolean indicating whether verbose messages should be
   * logged (true) or not (false)
   */
  ILAOStarSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, double discount = 1.0,
      double epsilon = 0.001,
      const CallbackFunctor &callback = [](const ILAOStarSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the ILAO* algorithm
   *
   * @param s Root state of the search from which ILAO* policy graph is computed
   */
  void solve(const State &s);

  /**
   * @brief Indicates whether the solution policy is defined for a given state
   *
   * @param s State for which an entry is searched in the policy graph
   * @return true If the state has been explored and an action is defined in
   * this state
   * @return false If the state has not been explored or no action is defined in
   * this state
   */
  bool is_solution_defined_for(const State &s) const;

  /**
   * @brief Get the best computed action in terms of best Q-value in a given
   * state (throws a runtime error exception if no action is defined in the
   * given state, which is why it is advised to call
   * ILAOStarSolver::is_solution_defined_for before).
   *
   * @param s State for which the best action is requested
   * @return const Action& Best computed action
   */
  const Action &get_best_action(const State &s) const;

  /**
   * @brief Get the best Q-value in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call ILAOStarSolver::is_solution_defined_for before)
   *
   * @param s State from which the best Q-value is requested
   * @return double Minimum Q-value of the given state over the applicable
   * actions in this state
   */
  Value get_best_value(const State &s) const;

  /**
   * @brief Get the number of states present in the search graph
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_explored_states() const;

  /**
   * @brief Get the set of states present in the search graph (i.e. the graph's
   * state nodes minus the nodes' encapsulation and their children)
   *
   * @return SetTypeDeducer<State>::Set Set of states present in the search
   * graph
   */
  typename SetTypeDeducer<State>::Set get_explored_states() const;

  /**
   * @brief Get the number of states present in the current best policy graph
   *
   * @return std::size_t Number of states present in the current best policy
   * graph
   */
  std::size_t best_solution_graph_size() const;

  /**
   * @brief Get the solving time in milliseconds since the beginning of the
   * search from the root solving state
   *
   * @return std::size_t Solving time in milliseconds
   */
  std::size_t get_solving_time() const;

  /**
   * @brief Get the (partial) solution policy defined for the states for which
   * the Q-value has been updated at least once (which is optimal for the
   * non-tip states reachable by this policy)
   *
   * @return Mapping from states to pairs of action and best Q-value
   */
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const;

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  atomic_double _discount;
  atomic_double _epsilon;
  CallbackFunctor _callback;
  bool _verbose;
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
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

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
