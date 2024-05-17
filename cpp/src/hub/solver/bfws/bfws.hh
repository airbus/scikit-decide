/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_BFWS_HH
#define SKDECIDE_BFWS_HH

// From paper: Best-First Width Search: Exploration and Exploitation in
// Classical Planning
//             by Nir Lipovetsky and Hector Geffner
//             in proceedings of AAAI 2017

#include <functional>
#include <memory>
#include <unordered_set>
#include <queue>
#include <list>
#include <chrono>

#include <boost/container_hash/hash.hpp>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

/** Use default hasher provided with domain's states */
template <typename Tdomain, typename Tfeature_vector> struct DomainStateHash {
  typedef typename Tdomain::State Key;

  template <typename Tnode> static const Key &get_key(const Tnode &n);

  struct Hash {
    std::size_t operator()(const Key &k) const;
  };

  struct Equal {
    bool operator()(const Key &k1, const Key &k2) const;
  };
};

/** Use state binary feature vector to hash states */
template <typename Tdomain, typename Tfeature_vector> struct StateFeatureHash {
  typedef Tfeature_vector Key;

  template <typename Tnode> static const Key &get_key(const Tnode &n);

  struct Hash {
    std::size_t operator()(const Key &k) const;
  };

  struct Equal {
    bool operator()(const Key &k1, const Key &k2) const;
  };
};

/**
 * @brief This is the skdecide implementation Best First Width Search from
 * "Best-First Width Search: Exploration and Exploitation in Classical Planning"
 * by Nir Lipovetzky and Hector Geffner (2017)
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Tfeature_vector Type of of the state feature vector used to compute
 * the novelty measure
 * @tparam Thashing_policy Type of the hashing class used to hash states (one of
 * 'DomainStateHash' to use the state hash function, or 'StateFeatureHash' to
 * hash states based on their features)
 * @tparam Texecution_policy Type of the execution policy (one of
 * 'SequentialExecution' to generate state-action transitions in sequence,
 * or 'ParallelExecution' to generate state-action transitions in parallel on
 * different threads)
 */
template <typename Tdomain, typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          typename Texecution_policy = SequentialExecution>
class BFWSSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Tfeature_vector FeatureVector;
  typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<std::unique_ptr<FeatureVector>(Domain &d,
                                                       const State &s)>
      StateFeatureFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<bool(const BFWSSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new BFWSSolver object
   *
   * @param domain The domain instance
   * @param goal_checker Functor taking as arguments the domain and a
   * state object, and returning true if the state is a goal
   * @param state_features State feature vector used to compute the novelty
   * measure
   * @param heuristic Functor taking as arguments the domain and a state object,
   * and returning the heuristic estimate from the state to the goal
   * @param callback Functor called before popping the next state from the
   * (priority) open queue, taking as arguments the solver and the domain, and
   * returning true if the solver must be stopped
   * @param verbose Boolean indicating whether verbose messages should be
   * logged (true) or not (false)
   */
  BFWSSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const StateFeatureFunctor &state_features,
      const HeuristicFunctor &heuristic,
      const CallbackFunctor &callback = [](const BFWSSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the BFWS algorithm
   *
   * @param s Root state of the search from which BFWS graph traversals are
   * performed
   */
  void solve(const State &s);

  /**
   * @brief Indicates whether the solution (potentially built from merging
   * several previously computed plans) is defined for a given state
   *
   * @param s State for which an entry is searched in the policy graph
   * @return true If a plan that goes through the state has been previously
   * computed
   * @return false If no plan that goes through the state has been previously
   * computed
   */
  bool is_solution_defined_for(const State &s) const;

  /**
   * @brief Get the best computed action in terms of minimum cost-to-go in a
   * given state (throws a runtime error exception if no action is defined in
   * the given state, which is why it is advised to call
   * BFWSSolver::is_solution_defined_for before).
   *
   * @param s State for which the best action is requested
   * @return const Action& Best computed action
   */
  const Action &get_best_action(const State &s) const;

  /**
   * @brief Get the minimum cost-to-go in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call BFWSSolver::is_solution_defined_for before)
   *
   * @param s State from which the minimum cost-to-go is requested
   * @return double Minimum cost-to-go of the given state over the applicable
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
   * state nodes minus the nodes' encapsulation and their neighbors)
   *
   * @return SetTypeDeducer<State>::Set Set of states present in the search
   * graph
   */
  typename SetTypeDeducer<State>::Set get_explored_states() const;

  /**
   * @brief Get the number of states present in the priority queue (i.e. those
   * explored states that have not been yet closed by BFWS)
   *
   * @return std::size_t Number of states present in the (priority) open queue
   */
  std::size_t get_nb_tip_states() const;

  /**
   * @brief Get the top tip state, i.e. the tip state with the lowest f-score
   *
   * @return const State& Next tip state to be closed by BFWS
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
   * @brief Get the solution plan starting in a given state (throws a runtime
   * exception if a state cycle is detected in the plan)
   *
   * @param from_state State from which a solution plan to a goal state is
   * requested
   * @return std::vector<std::tuple<State, Action, Value>> Sequence of tuples of
   * state, action and transition cost (computed as the difference of g-scores
   * between this state and the next one) visited along the execution of the
   * plan; or an empty plan if no plan was previously computed that goes through
   * the given state.
   */
  std::vector<std::tuple<State, Action, Value>>
  get_plan(const State &from_state) const;

  /**
   * @brief Get the (partial) solution policy defined for the states for which
   * a solution plan that goes through them has been previously computed at
   * least once
   *
   * @return Mapping from states to pairs of action and minimum cost-to-go
   */
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map
  get_policy() const;

private:
  Domain &_domain;
  StateFeatureFunctor _state_features;
  HeuristicFunctor _heuristic;
  GoalCheckerFunctor _goal_checker;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  typedef std::pair<std::size_t, typename FeatureVector::value_type> PairType;
  typedef std::unordered_map<
      double, std::unordered_set<PairType, boost::hash<PairType>>>
      PairMap;

  struct Node {
    State state;
    std::unique_ptr<FeatureVector> features;
    std::tuple<Node *, Action, double> best_parent;
    std::size_t novelty;
    double heuristic;
    double gscore;
    double fscore;
    std::pair<Action *, Node *>
        best_action; // computed only when constructing the solution path
                     // backward from the goal state
    bool solved;     // set to true if on the solution path constructed backward
                     // from the goal state

    Node(const State &s, Domain &d,
         const std::function<std::unique_ptr<FeatureVector>(
             Domain &d, const State &s)> &state_features);

    struct Key {
      const typename HashingPolicy::Key &operator()(const Node &n) const;
    };
  };

  // we only compute novelty of 1 for complexity reasons and assign all other
  // novelties to +infty see paper "Best-First Width Search: Exploration and
  // Exploitation in Classical Planning" by Lipovetsky and Geffner
  std::size_t novelty(PairMap &heuristic_features_map,
                      const double &heuristic_value, Node &n) const;

  struct NodeCompare {
    bool operator()(Node *&a, Node *&b) const;
  };

  typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
  Graph _graph;

  typedef std::priority_queue<Node *, std::vector<Node *>, NodeCompare>
      PriorityQueue;
  PriorityQueue _open_queue;

  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

} // namespace skdecide

#endif // SKDECIDE_BFWS_HH
