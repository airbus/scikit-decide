/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_ASTAR_HH
#define SKDECIDE_ASTAR_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <queue>
#include <list>
#include <chrono>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

template <typename Tdomain,
          typename Texecution_policy = SequentialExecution>
class AStarSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef typename Domain::Predicate Predicate;
    typedef typename Domain::Value Value;
    typedef Texecution_policy ExecutionPolicy;

    AStarSolver(Domain& domain,
                const std::function<Predicate (Domain&, const State&)>& goal_checker,
                const std::function<Value (Domain&, const State&)>& heuristic,
                bool debug_logs = false);
    
    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear();
    
    // solves from state s using heuristic function h
    void solve(const State& s);

    bool is_solution_defined_for(const State& s) const;
    const Action& get_best_action(const State& s) const;
    const double& get_best_value(const State& s) const;

private :
    Domain& _domain;
    std::function<bool (Domain&, const State&)> _goal_checker;
    std::function<Value (Domain&, const State&)> _heuristic;
    bool _debug_logs;
    ExecutionPolicy _execution_policy;

    struct Node {
        State state;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore;
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool solved; // set to true if on the solution path constructed backward from the goal state

        Node(const State& s);
        
        struct Key {
            const State& operator()(const Node& sn) const;
        };
    };

    struct NodeCompare {
        bool operator()(Node*& a, Node*& b) const;
    };

    typedef typename SetTypeDeducer<Node, State>::Set Graph;
    Graph _graph;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/astar_impl.hh"
#endif

#endif // SKDECIDE_ASTAR_HH
