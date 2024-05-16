/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_IW_HH
#define SKDECIDE_IW_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
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
 * @brief This is the skdecide implementation of the Iterated Width as described
 * in "Width and Serialization of Classical Planning Problems" by Nir Lipovetzky
 * and Hector Geffner (2012)
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
class IWSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef Tfeature_vector FeatureVector;
  typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<std::unique_ptr<FeatureVector>(Domain &d,
                                                       const State &s)>
      StateFeatureFunctor;
  typedef std::function<bool(const double &, const std::size_t &,
                             const std::size_t &, const double &,
                             const std::size_t &, const std::size_t &)>
      NodeOrderingFunctor;
  typedef std::function<bool(const IWSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new IWSolver object
   *
   * @param domain domain The domain instance
   * @param state_features State feature vector used to compute the novelty
   * measure
   * @param node_ordering Functor called to rank two search nodes A and B,
   * taking as inputs A's g-score, A's novelty, A's search depth, B's g-score,
   * B's novelty, B's search depth, and returning true when B should be
   * preferred to A (defaults to rank nodes based on their g-scores)
   * @param time_budget Maximum time allowed (in milliseconds) to continue
   * searching for better plans after a first plan reaching a goal has been
   * found
   * @param callback Functor called before popping the next state from the
   * (priority) open queue, taking as arguments the solver and the domain, and
   * returning true if the solver must be stopped
   * @param verbose Boolean indicating whether verbose messages should be
   * logged (true) or not (false)
   */
  IWSolver(
      Domain &domain, const StateFeatureFunctor &state_features,
      const NodeOrderingFunctor &node_ordering =
          [](const double &a_gscore, const std::size_t &a_novelty,
             const std::size_t &a_depth, const double &b_gscore,
             const std::size_t &b_novelty, const std::size_t &b_depth) -> bool {
        return a_gscore > b_gscore;
      },
      std::size_t time_budget = 0, // time budget to continue searching for
                                   // better plans after a goal has been reached
      const CallbackFunctor &callback = [](const IWSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the IW algorithm
   *
   * @param s Root state of the search from which IW graph traversals are
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
   * IWSolver::is_solution_defined_for before).
   *
   * @param s State for which the best action is requested
   * @return const Action& Best computed action
   */
  const Action &get_best_action(const State &s) const;

  /**
   * @brief Get the minimum cost-to-go in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call IWSolver::is_solution_defined_for before)
   *
   * @param s State from which the minimum cost-to-go is requested
   * @return double Minimum cost-to-go of the given state over the applicable
   * actions in this state
   */
  Value get_best_value(const State &s) const;

  /**
   * @brief Get the current width of the search (or final width of the domain if
   * the method has not been called from the callback functor)
   *
   * @return const std::size_t& Current width of the search
   */
  const std::size_t &get_current_width() const;

  /**
   * @brief Get the number of states present in the search graph
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_of_explored_states() const;

  /**
   * @brief Get the set of states present in the search graph (i.e. the graph's
   * state nodes minus the nodes' encapsulation and their neighbors)
   *
   * @return SetTypeDeducer<State>::Set Set of states present in the search
   * graph
   */
  typename SetTypeDeducer<State>::Set get_explored_states() const;

  /**
   * @brief Get the number of states pruned by the novelty measure among the
   * ones present in the search graph
   *
   * @return SetTypeDeducer<State>::Set Number of states pruned by the novelty
   * measure among the ones present in the search graph graph
   */
  std::size_t get_nb_of_pruned_states() const;

  /**
   * @brief Get the number of states present in the priority queue (i.e. those
   * explored states that have not been yet closed by IW) of the current width
   * search procedure (throws a runtime exception if no active width sub-solver
   * is active)
   *
   * @return std::size_t Number of states present in the (priority) open queue
   * of the current width search procedure
   */
  std::size_t get_nb_tip_states() const;

  /**
   * @brief Get the top tip state, i.e. the tip state with the lowest
   * lexicographical score (according to the node ordering functor given in the
   * IWSolver instance's constructor) of the current width search procedure
   * (throws a runtime exception if no active width sub-solver is active)
   *
   * @return const State& Next tip state to be closed by the current width
   * search procedure
   */
  const State &get_top_tip_state() const;

  /**
   * @brief Get the history of tuples of time point (in milliseconds), current
   * width, and root state's f-score, recorded each time a goal state is
   * encountered during the search
   *
   * @return const std::list<std::tuple<std::size_t, std::size_t, double>>&
   * List of tuples of time point (in milliseconds), current width, and root
   * state's f-score
   */
  const std::list<std::tuple<std::size_t, std::size_t, double>> &
  get_intermediate_scores() const;

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
  NodeOrderingFunctor _node_ordering;
  std::size_t _time_budget;
  std::list<std::tuple<std::size_t, std::size_t, double>> _intermediate_scores;
  CallbackFunctor _callback;
  bool _verbose;

  struct Node {
    State state;
    std::unique_ptr<FeatureVector> features;
    std::tuple<Node *, Action, double> best_parent;
    double gscore;
    double fscore; // not in A*'s meaning but rather to store cost-to-go once a
                   // solution is found
    std::size_t novelty;
    std::size_t depth;
    std::pair<Action *, Node *>
        best_action; // computed only when constructing the solution path
                     // backward from the goal state
    bool pruned; // true if pruned by the novelty test (used only to report nb
                 // of pruned states)
    bool solved; // set to true if on the solution path constructed backward
                 // from the goal state

    Node(const State &s, Domain &d,
         const std::function<std::unique_ptr<FeatureVector>(
             Domain &d, const State &s)> &state_features);

    struct Key {
      const typename HashingPolicy::Key &operator()(const Node &n) const;
    };
  };

  typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
  Graph _graph;

  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  class WidthSolver { // known as IW(i), i.e. the fixed-width solver
                      // sequentially run by IW
  public:
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    typedef Texecution_policy ExecutionPolicy;

    WidthSolver(IWSolver &parent_solver, Domain &domain,
                const StateFeatureFunctor &state_features,
                const NodeOrderingFunctor &node_ordering, std::size_t width,
                Graph &graph, std::size_t time_budget,
                std::list<std::tuple<std::size_t, std::size_t, double>>
                    &intermediate_scores,
                const CallbackFunctor &callback, bool verbose);

    // solves from state s
    // returned pair p: p.first==true iff solvable, p.second==true iff states
    // have been pruned
    std::pair<bool, bool>
    solve(const State &s,
          const std::chrono::time_point<std::chrono::high_resolution_clock>
              &start_time,
          bool &found_goal);

  private:
    IWSolver &_parent_solver;
    Domain &_domain;
    const StateFeatureFunctor &_state_features;
    const NodeOrderingFunctor &_node_ordering;
    std::size_t _width;
    Graph &_graph;
    std::size_t _time_budget;
    std::list<std::tuple<std::size_t, std::size_t, double>>
        &_intermediate_scores;
    const CallbackFunctor &_callback;
    bool _verbose;
    ExecutionPolicy _execution_policy;
    typedef std::vector<
        std::pair<std::size_t, typename FeatureVector::value_type>>
        TupleType;
    typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>>
        TupleVector;

    struct NodeCompare {
      NodeCompare(const NodeOrderingFunctor &node_ordering);
      bool operator()(Node *&a, Node *&b) const;
      const NodeOrderingFunctor &_node_ordering;
    };

  public:
    typedef std::priority_queue<Node *, std::vector<Node *>, NodeCompare>
        PriorityQueue;

    const PriorityQueue &get_open_queue() const;

  private:
    std::unique_ptr<PriorityQueue> _open_queue;

    std::size_t novelty(TupleVector &feature_tuples, Node &n) const;

    // Generates all combinations of size k from [0 ... (n-1)]
    void generate_tuples(const std::size_t &k, const std::size_t &n,
                         const std::function<void(TupleType &)> &f) const;
  };

  std::size_t _width;
  std::unique_ptr<WidthSolver> _width_solver;
};

} // namespace skdecide

#endif // SKDECIDE_IW_HH
