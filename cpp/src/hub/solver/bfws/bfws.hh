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

template <typename Tdomain, typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          typename Texecution_policy = SequentialExecution>
class BFWSSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef Tfeature_vector FeatureVector;
  typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
  typedef Texecution_policy ExecutionPolicy;

  BFWSSolver(
      Domain &domain,
      const std::function<std::unique_ptr<FeatureVector>(
          Domain &d, const State &s)> &state_features,
      const std::function<Value(Domain &, const State &)> &heuristic,
      const std::function<bool(Domain &, const State &)> &termination_checker,
      bool debug_logs = false);

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  const double &get_best_value(const State &s) const;

private:
  Domain &_domain;
  std::function<std::unique_ptr<FeatureVector>(Domain &d, const State &s)>
      _state_features;
  std::function<Value(Domain &, const State &)> _heuristic;
  std::function<bool(Domain &, const State &)> _termination_checker;
  bool _debug_logs;
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
    Action *best_action; // computed only when constructing the solution path
                         // backward from the goal state
    bool solved; // set to true if on the solution path constructed backward
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
};

} // namespace skdecide

#endif // SKDECIDE_BFWS_HH
