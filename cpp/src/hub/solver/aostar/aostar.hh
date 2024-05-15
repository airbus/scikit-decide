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
 * or 'ParallelExecution' to generate state-action transitions in parallel on
 * different threads)
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

  /**
   * @brief Construct a new AOStarSolver object
   *
   * @param domain The domain instance
   * @param goal_checker Functor taking as arguments the domain and a state
   * object, and returning true if the state is the goal
   * @param heuristic Functor taking as arguments the domain and a state object,
   * and returning the heuristic estimate from the state to the goal
   * @param discount Value function's discount factor
   * @param max_tip_expansions Maximum number of states to extract from the
   * priority queue at each iteration before recomputing the policy graph
   * @param detect_cycles Boolean indicating whether cycles in the search graph
   * should be automatically detected (true) or not (false), knowing that the
   * AO* algorithm is not meant to work with graph cycles into which it might be
   * infinitely trapped
   * @param debug_logs Boolean indicating whether debugging messages should be
   * logged (true) or not (false)
   * @param callback Functor called before popping the next state from the
   * priority queue, taking as arguments the solver and the domain, and
   * returning true if the solver must be stopped
   */
  AOStarSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, double discount = 1.0,
      std::size_t max_tip_expansions = 1, bool detect_cycles = false,
      bool debug_logs = false,
      const CallbackFunctor &callback = [](const AOStarSolver &, Domain &) {
        return false;
      });

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the AO* algorithm
   *
   * @param s Root state of the search from which AO* graph traversals are
   * performed
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
   * AOStarSolver::is_solution_defined_for before).
   *
   * @param s State for which the best action is requested
   * @return const Action& Best computed action
   */
  const Action &get_best_action(const State &s) const;

  /**
   * @brief Get the best Q-value in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call AOStarSolver::is_solution_defined_for before)
   *
   * @param s State from which the best Q-value is requested
   * @return double Maximum Q-value of the given state over the applicable
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
   * @brief Get the number of states present in the priority queue (i.e. those
   * explored states that have not been yet expanded)
   *
   * @return std::size_t Number of states present in the priority queue
   */
  std::size_t get_nb_tip_states() const;

  /**
   * @brief Get the top tip state, i.e. the tip state with the lowest value
   * function
   *
   * @return const State& Next tip state to be expanded by the algorithm
   */
  const State &get_top_tip_state() const;

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
   * non-tip states reachable by this policy
   *
   * @return Mapping from states to pairs of action and best Q-value
   */
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
