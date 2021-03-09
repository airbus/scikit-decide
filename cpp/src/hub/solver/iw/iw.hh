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
template <typename Tdomain, typename Tfeature_vector>
struct DomainStateHash {
    typedef typename Tdomain::State Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n);

    struct Hash {
        std::size_t operator()(const Key& k) const;
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const;
    };
};


/** Use state binary feature vector to hash states */
template <typename Tdomain, typename Tfeature_vector>
struct StateFeatureHash {
    typedef Tfeature_vector Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n);

    struct Hash {
        std::size_t operator()(const Key& k) const;
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const;
    };
};


template <typename Tdomain,
          typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          typename Texecution_policy = SequentialExecution>
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    typedef Texecution_policy ExecutionPolicy;

    IWSolver(Domain& domain,
             const std::function<std::unique_ptr<FeatureVector> (Domain& d, const State& s)>& state_features,
             const std::function<bool (const double&, const std::size_t&, const std::size_t&,
                                       const double&, const std::size_t&, const std::size_t&)>& node_ordering = nullptr,
             std::size_t time_budget = 0,  // time budget to continue searching for better plans after a goal has been reached
             bool debug_logs = false);

    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear();

    // solves from state s
    void solve(const State& s);

    bool is_solution_defined_for(const State& s) const;
    const Action& get_best_action(const State& s) const;
    const double& get_best_value(const State& s) const;
    std::size_t get_nb_of_explored_states() const;
    std::size_t get_nb_of_pruned_states() const;
    const std::list<std::tuple<std::size_t, std::size_t, double>>& get_intermediate_scores() const;

private :

    Domain& _domain;
    std::function<std::unique_ptr<FeatureVector> (Domain& d, const State& s)> _state_features;
    std::function<bool (const double&, const std::size_t&, const std::size_t&,
                                       const double&, const std::size_t&, const std::size_t&)> _node_ordering;
    std::size_t _time_budget;
    std::list<std::tuple<std::size_t, std::size_t, double>> _intermediate_scores;
    bool _debug_logs;

    struct Node {
        State state;
        std::unique_ptr<FeatureVector> features;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore; // not in A*'s meaning but rather to store cost-to-go once a solution is found
        std::size_t novelty;
        std::size_t depth;
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool pruned; // true if pruned by the novelty test (used only to report nb of pruned states)
        bool solved; // set to true if on the solution path constructed backward from the goal state

        Node(const State& s, Domain& d,
             const std::function<std::unique_ptr<FeatureVector> (Domain& d, const State& s)>& state_features);
        
        struct Key {
            const typename HashingPolicy::Key& operator()(const Node& n) const;
        };
    };

    typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
    Graph _graph;

    class WidthSolver { // known as IW(i), i.e. the fixed-width solver sequentially run by IW
    public :
        typedef Tdomain Domain;
        typedef typename Domain::State State;
        typedef typename Domain::Action Action;
        typedef Tfeature_vector FeatureVector;
        typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
        typedef Texecution_policy ExecutionPolicy;

        WidthSolver(Domain& domain,
                    const std::function<std::unique_ptr<FeatureVector> (Domain& d, const State& s)>& state_features,
                    const std::function<bool (const double&, const std::size_t&, const std::size_t&,
                                              const double&, const std::size_t&, const std::size_t&)>& node_ordering,
                    std::size_t width,
                    Graph& graph,
                    std::size_t time_budget,
                    std::list<std::tuple<std::size_t, std::size_t, double>>& intermediate_scores,
                    bool debug_logs);
        
        // solves from state s
        // returned pair p: p.first==true iff solvable, p.second==true iff states have been pruned
        std::pair<bool, bool> solve(const State& s, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time, bool& found_goal);

    private :
        Domain& _domain;
        const std::function<std::unique_ptr<FeatureVector> (Domain& d, const State& s)>& _state_features;
        const std::function<bool (const double&, const std::size_t&, const std::size_t&,
                                  const double&, const std::size_t&, const std::size_t&)>& _node_ordering;
        std::size_t _width;
        Graph& _graph;
        std::size_t _time_budget;
        std::list<std::tuple<std::size_t, std::size_t, double>>& _intermediate_scores;
        bool _debug_logs;
        ExecutionPolicy _execution_policy;

        typedef std::vector<std::pair<std::size_t, typename FeatureVector::value_type>> TupleType;
        typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>> TupleVector;

        std::size_t novelty(TupleVector& feature_tuples, Node& n) const;

        // Generates all combinations of size k from [0 ... (n-1)]
        void generate_tuples(const std::size_t& k,
                             const std::size_t& n,
                             const std::function<void (TupleType&)>& f) const;
    };
};

} // namespace skdecide

#endif // SKDECIDE_IW_HH
