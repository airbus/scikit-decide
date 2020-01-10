/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_BFWS_HH
#define SKDECIDE_BFWS_HH

// From paper: Best-First Width Search: Exploration and Exploitation in Classical Planning
//             by Nir Lipovetsky and Hector Geffner
//             in proceedings of AAAI 2017

#include <functional>
#include <memory>
#include <unordered_set>
#include <queue>
#include <list>
#include <chrono>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class BFWSSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Texecution_policy ExecutionPolicy;

    BFWSSolver(Domain& domain,
               const std::function<void (const State&, const std::function<void (const unsigned int&)>&)>& state_binarizer,
               const std::function<double (const State&)>& heuristic,
               const std::function<bool (const State&)>& termination_checker,
               bool debug_logs = false)
        : _domain(domain), _state_binarizer(state_binarizer), _heuristic(heuristic), _termination_checker(termination_checker),
          _debug_logs(debug_logs) {
              if (debug_logs) {
                spdlog::set_level(spdlog::level::debug);
              } else {
                  spdlog::set_level(spdlog::level::info);
              }
          }
    
    // clears the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void clear() {
        _graph.clear();
    }
    
    // solves from state s
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print() + " BFWS solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();

            // Map from heuristic values to set of state bits with that given heuristic value
            // that have been true at least once since the beginning of the search (stored by their index)
            std::unordered_map<double, std::unordered_set<unsigned int>> true_bits_map;

            // Create the root node containing the given state s
            auto si = _graph.emplace(s);
            if (si.first->solved || _termination_checker(s)) { // problem already solved from this state (was present in _graph and already solved)
                return;
            }
            Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
            root_node.gscore = 0;
            root_node.heuristic = _heuristic(root_node.state);
            root_node.novelty = novelty(true_bits_map, root_node.heuristic, root_node.state);

            // Priority queue used to sort non-goal unsolved tip nodes by increasing cost-to-go values (so-called OPEN container)
            std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_queue;
            open_queue.push(&root_node);

            // Set of states that have already been explored
            std::unordered_set<Node*> closed_set;

            while (!open_queue.empty()) {
                auto best_tip_node = open_queue.top();
                open_queue.pop();

                // Check that the best tip node has not already been closed before
                // (since this implementation's open_queue does not check for element uniqueness,
                // it can contain many copies of the same node pointer that could have been closed earlier)
                if (closed_set.find(best_tip_node) != closed_set.end()) { // this implementation's open_queue can contain several
                    continue;
                }

                if (_debug_logs) spdlog::debug("Current best tip node (h=" + std::to_string(best_tip_node->heuristic) +
                                                                      ", n=" + std::to_string(best_tip_node->novelty) +
                                                                      "): " + best_tip_node->state.print());

                if (_termination_checker(best_tip_node->state) || best_tip_node->solved) {
                    if (_debug_logs) spdlog::debug("Found a terminal state: " + best_tip_node->state.print());
                    auto current_node = best_tip_node;
                    if (!(best_tip_node->solved)) { current_node->fscore = 0; } // goal state

                    while (current_node != &root_node) {
                        Node* parent_node = std::get<0>(current_node->best_parent);
                        parent_node->best_action = &std::get<1>(current_node->best_parent);
                        parent_node->fscore = std::get<2>(current_node->best_parent) + current_node->fscore;
                        parent_node->solved = true;
                        current_node = parent_node;
                    }

                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                    spdlog::info("BFWS finished to solve from state " + s.print() + " in " + std::to_string((double) duration / (double) 1e9) + " seconds.");
                    return;
                }

                closed_set.insert(best_tip_node);

                // Expand best tip node
                auto applicable_actions = _domain.get_applicable_actions(best_tip_node->state)->get_elements();
                std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node](const auto& a){
                    if (_debug_logs) spdlog::debug("Current expanded action: " + Action(a).print());
                    // Asynchronously compute next state distribution
                    // Must be separated from next loop in case the domain is python so that it is in this case actually implemented as a pool of independent processes
                    _domain.compute_next_state(best_tip_node->state, a);
                });
                std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node, &open_queue, &closed_set, &true_bits_map](const auto& a){
                    auto next_state = _domain.get_next_state(best_tip_node->state, a);
                    std::pair<typename Graph::iterator, bool> i;
                    _execution_policy.protect([this, &i, &next_state]{
                        i = _graph.emplace(next_state);
                    });
                    Node& neighbor = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                    if (_debug_logs) spdlog::debug("Exploring next state: " + neighbor.state.print());

                    if (closed_set.find(&neighbor) != closed_set.end()) {
                        // Ignore the neighbor which is already evaluated
                        return;
                    }

                    double transition_cost = _domain.get_transition_cost(best_tip_node->state, a, neighbor.state);
                    double tentative_gscore = best_tip_node->gscore + transition_cost;

                    if ((i.second) || (tentative_gscore < neighbor.gscore)) {
                        if (_debug_logs) spdlog::debug("New gscore: " + std::to_string(best_tip_node->gscore) + "+" +
                                                        std::to_string(transition_cost) + "=" + std::to_string(tentative_gscore));
                        neighbor.gscore = tentative_gscore;
                        neighbor.best_parent = std::make_tuple(best_tip_node, a, transition_cost);
                    }

                    neighbor.heuristic = _heuristic(neighbor.state);
                    if (_debug_logs) spdlog::debug("Heuristic: " + std::to_string(neighbor.heuristic));
                    _execution_policy.protect([this, &true_bits_map, &open_queue, &neighbor]{
                        neighbor.novelty = novelty(true_bits_map, neighbor.heuristic, neighbor.state);
                        open_queue.push(&neighbor);
                        if (_debug_logs) spdlog::debug("Novelty: " + std::to_string(neighbor.novelty));
                    });
                });
            }

            spdlog::info("BFWS could not find a solution from state " + s.print());
        } catch (const std::exception& e) {
            spdlog::error("BFWS failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr) || (si->solved == false)) {
            return false;
        } else {
            return true;
        }
    }

    const Action& get_best_action(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        }
        return *(si->best_action);
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            throw std::runtime_error("SKDECIDE exception: no best action found in state " + s.print());
        }
        return si->fscore;
    }

private :
    Domain& _domain;
    std::function<void (const State&, const std::function<void (const unsigned int&)>&)> _state_binarizer;
    std::function<double (const State&)> _heuristic;
    std::function<bool (const State&)> _termination_checker;
    bool _debug_logs;
    ExecutionPolicy _execution_policy;

    // we only compute novelty of 1 for complexity reasons and assign all other novelties to +infty
    // see paper "Best-First Width Search: Exploration and Exploitation in Classical Planning" by Lipovetsky and Geffner
    unsigned int novelty(std::unordered_map<double, std::unordered_set<unsigned int>>& true_bits_map,
                         const double& heuristic_value, const State& state) const {
        auto r = true_bits_map.emplace(heuristic_value, std::unordered_set<unsigned int>());
        std::unordered_set<unsigned int>& true_bits = r.first->second;
        unsigned int nov = 0;
        _state_binarizer(state, [&nov, &true_bits](const unsigned int& i){
            nov += (int) true_bits.insert(i).second;
        });
        if (r.second) {
            nov = 0;
        } else if (nov == 0) {
            nov = std::numeric_limits<unsigned int>::max();
        }
        return nov;
    }

    struct Node {
        State state;
        std::tuple<Node*, Action, double> best_parent;
        unsigned int novelty;
        double heuristic;
        double gscore;
        double fscore;
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool solved; // set to true if on the solution path constructed backward from the goal state

        Node(const State& s)
            : state(s),
              novelty(std::numeric_limits<unsigned int>::max()),
              gscore(std::numeric_limits<double>::infinity()),
              fscore(std::numeric_limits<double>::infinity()),
              best_action(nullptr),
              solved(false) {}
        
        struct Key {
            const State& operator()(const Node& sn) const { return sn.state; }
        };
    };

    struct NodeCompare {
        bool operator()(Node*& a, Node*& b) const {
            // smallest element appears at the top of the priority_queue => cost optimization
            // rank first by heuristic values then by novelty measures
            return ((a->heuristic) > (b->heuristic)) ||
                    (((a->heuristic) == (b->heuristic)) && ((a->novelty) > (b->novelty)));
        }
    };

    typedef typename SetTypeDeducer<Node, State>::Set Graph;
    Graph _graph;
};

} // namespace skdecide

#endif // SKDECIDE_BFWS_HH
