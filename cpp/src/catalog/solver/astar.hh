#ifndef AIRLAPS_ASTAR_HH
#define AIRLAPS_ASTAR_HH

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

namespace airlaps {

template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class AStarSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Texecution_policy ExecutionPolicy;

    AStarSolver(Domain& domain,
                const std::function<bool (const State&)>& goal_checker,
                const std::function<double (const State&)>& heuristic,
                bool debug_logs = false)
        : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
          _debug_logs(debug_logs) {
              if (debug_logs) {
                spdlog::set_level(spdlog::level::debug);
              } else {
                  spdlog::set_level(spdlog::level::info);
              }
          }
    
    // solves from state s using heuristic function h
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print() + " A* solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();

            // Create the root node containing the given state s
            auto si = _graph.emplace(s);
            if (si.first->solved || _goal_checker(s)) { // problem already solved from this state (was present in _graph and already solved)
                return;
            }
            Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
            root_node.best_value = _heuristic(root_node.state);

            // Priority queue used to sort non-goal unsolved tip nodes by increasing cost-to-go values (so-called OPEN container)
            std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_queue;
            open_queue.push(&root_node);

            // Set of states for which the g-value is optimal (so-called CLOSED container)
            typedef typename SetTypeDeducer<Node*, State>::Set closed_set;

            while (!open_queue.empty()) {
                auto best_tip_node = open_queue.top();
                open_queue.pop();

                // Check that the best tip node has not already been closed before
                // (since this implementation's open_queue does not check for element uniqueness,
                // it can contain many copies of the same node pointer that could have been closed earlier)
                if (closed_set.find(best_tip_node) != closed_set.end()) { // this implementation's open_queue can contain several
                    continue;
                }

                if (_debug_logs) spdlog::debug("Current best tip node: " + best_tip_node->state.print());

                if (_goal_checker(best_tip_node->state)) {
                    if (_debug_logs) spdlog::debug("Closing a goal state: " + best_tip_node->state.print());
                    auto current_node = best_tip_node;
                    current_node->fscore = 0;

                    while (current_node != &root_node) {
                        Node* parent_node = std::get<0>(current_node->best_parent);
                        parent_node->best_action = std::get<1>(&(current_node->best_parent));
                        parent_node->fscore = std::get<2>(current_node.best_parent) + current_node->fscore;
                        parent_node->solved = true;
                        current_node = parent_node;
                    }

                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                    spdlog::info("A* finished to solve from state " + s.print() + " in " + std::to_string((double) duration / (double) 1e9) + " seconds.");
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
                std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node](const auto& a){
                    auto next_state = _domain.get_next_state(best_tip_node->state, a);
                    std::pair<typename Graph::iterator, bool> i;
                    _execution_policy.protect([this, &i, &next_state]{
                        i = _graph.emplace(next_state);
                    });
                    Node& neighbor = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe

                    if (closed_set.find(&neighbor) != closed_set.end()) {
                        // Ignore the neighbor which is already evaluated
                        continue;
                    }

                    double transition_cost = _domain.get_transition_value(best_tip_node->state, a, neighbor.state);
                    double tentative_gscore = best_tip_node->gscore + transition_cost;

                    if ((i.second) || (tentative_gscore < neighbor.gscore)) {
                        neighbor.gscore = tentative_gscore;
                        neighbor.fscore = tentative_gscore + _heuristic(neighbor.state);
                        neighbor.best_parent = std::make_tuple(best_tip_node, a, transition_cost);
                        _execution_policy.protect([&open_queue, &neighbor]{
                            open_queue.push(neighbor);
                        }
                    }
                });
            }

            spdlog::info("A* could not find a solution from state " + s.print());
        } catch (const std::exception& e) {
            spdlog::error("A* failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    const Action& get_best_action(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return *(si->best_action);
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->fscore;
    }

private :
    Domain& _domain;
    std::function<bool (const State&)> _goal_checker;
    std::function<double (const State&)> _heuristic;
    bool _debug_logs;
    ExecutionPolicy _execution_policy;

    struct Node {
        State state;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore;
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool solved; // set to true if on the solution path constructed backward from the goal state

        Node(const State& s)
            : state(s),
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
            return (a->fscore) > (b->fscore); // smallest element appears at the top of the priority_queue => cost optimization
        }
    };

    typedef typename SetTypeDeducer<Node, State>::Set Graph;
    Graph _graph;
};

} // namespace airlaps

#endif // AIRLAPS_ASTAR_HH
