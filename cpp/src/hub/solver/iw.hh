#ifndef AIRLAPS_IW_HH
#define AIRLAPS_IW_HH

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
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Texecution_policy ExecutionPolicy;

    IWSolver(Domain& domain,
             const std::function<void (const State&, const std::function<void (const unsigned int&, const bool&)>&)>& state_binarizer,
             const std::function<bool (const State&)>& termination_checker,
             bool debug_logs = false)
    : _domain(domain), _state_binarizer(state_binarizer), _termination_checker(termination_checker),
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
            spdlog::info("Running " + ExecutionPolicy::print() + " IW solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();

            // Binarize state s to get number of bits
            unsigned int number_of_bits = 0;
            _state_binarizer(s, [&number_of_bits](const unsigned int& i, const bool& b){ number_of_bits++; });

            for (unsigned int w = 1 ; w <= number_of_bits ; w++) {
                std::pair<bool, bool> res = WidthSolver(_domain, w, _state_binarizer, _termination_checker, _graph, _debug_logs).solve(s);
                if (res.first) { // solution found with width w
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                    spdlog::info("IW finished to solve from state " + s.print() + " in " + std::to_string((double) duration / (double) 1e9) + " seconds.");
                    return;
                } else if (!res.second) { // no states pruned => problem is unsolvable
                    break;
                }
            }

            spdlog::info("IW could not find a solution from state " + s.print());
        } catch (const std::exception& e) {
            spdlog::error("IW failed solving from state " + s.print() + ". Reason: " + e.what());
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
    std::function<void (const State&, const std::function<void (const unsigned int&, const bool&)>&)> _state_binarizer;
    std::function<bool (const State&)> _termination_checker;
    bool _debug_logs;

    struct Node {
        State state;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore; // not in A*'s meaning but rather to store cost-to-go once a solution is found
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
            return (a->gscore) > (b->gscore); // smallest element appears at the top of the priority_queue => cost optimization
        }
    };

    typedef typename SetTypeDeducer<Node, State>::Set Graph;
    Graph _graph;

    class WidthSolver { // known as IW(i), i.e. the fixed-width solver sequentially run by IW
    public :
        typedef Tdomain Domain;
        typedef typename Domain::State State;
        typedef typename Domain::Event Action;
        typedef Texecution_policy ExecutionPolicy;

        WidthSolver(Domain& domain,
                    unsigned int width,
                    const std::function<void (const State&, const std::function<void (const unsigned int&, const bool&)>&)> state_binarizer,
                    const std::function<bool (const State&)>& termination_checker,
                    Graph& graph,
                    bool debug_logs = false)
            : _domain(domain), _width(width), _state_binarizer(state_binarizer), _termination_checker(termination_checker),
              _graph(graph), _debug_logs(debug_logs) {}
        
        // solves from state s
        // returned pair p: p.first==true iff solvable, p.second==true iff states have been pruned
        std::pair<bool, bool> solve(const State& s) {
            try {
                spdlog::info("Running " + ExecutionPolicy::print() + " IW(" + std::to_string(_width) + ") solver from state " + s.print());
                auto start_time = std::chrono::high_resolution_clock::now();

                // Create the root node containing the given state s
                auto si = _graph.emplace(s);
                if (si.first->solved || _termination_checker(s)) { // problem already solved from this state (was present in _graph and already solved)
                    return std::make_pair(true, false);
                }
                Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
                root_node.gscore = 0;
                bool states_pruned = false;

                // Priority queue used to sort non-goal unsolved tip nodes by increasing cost-to-go values (so-called OPEN container)
                std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_queue;
                open_queue.push(&root_node);

                // Set of state bits that have been true at least once since the beginning of the search (stored by their index)
                std::unordered_set<unsigned int> true_bits;
                novelty(true_bits, s); // we don't use the novelty number here, it's just to initialize true_bits with the root node's bits

                while (!open_queue.empty()) {
                    auto best_tip_node = open_queue.top();
                    open_queue.pop();

                    if (_debug_logs) spdlog::debug("Current best tip node: " + best_tip_node->state.print());

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
                        spdlog::info("IW(" + std::to_string(_width) + ") finished to solve from state " + s.print() + " in " + std::to_string((double) duration / (double) 1e9) + " seconds.");
                        return std::make_pair(true, states_pruned);
                    }

                    // Expand best tip node
                    auto applicable_actions = _domain.get_applicable_actions(best_tip_node->state)->get_elements();
                    std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node](const auto& a){
                        if (_debug_logs) spdlog::debug("Current expanded action: " + Action(a).print());
                        // Asynchronously compute next state distribution
                        // Must be separated from next loop in case the domain is python so that it is in this case actually implemented as a pool of independent processes
                        _domain.compute_next_state(best_tip_node->state, a);
                    });
                    std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node, &open_queue, &true_bits, &states_pruned](const auto& a){
                        auto next_state = _domain.get_next_state(best_tip_node->state, a);
                        std::pair<typename Graph::iterator, bool> i;
                        _execution_policy.protect([this, &i, &next_state]{
                            i = _graph.emplace(next_state);
                        });
                        Node& neighbor = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe

                        double transition_cost = _domain.get_transition_value(best_tip_node->state, a, neighbor.state);
                        double tentative_gscore = best_tip_node->gscore + transition_cost;

                        if ((i.second) || (tentative_gscore < neighbor.gscore)) {
                            neighbor.gscore = tentative_gscore;
                            neighbor.best_parent = std::make_tuple(best_tip_node, a, transition_cost);
                        }

                        if (novelty(true_bits, neighbor.state) > _width) {
                            states_pruned = true;
                        } else {
                            _execution_policy.protect([&open_queue, &neighbor]{
                                open_queue.push(&neighbor);
                            });
                        }
                    });
                }

                spdlog::info("IW(" + std::to_string(_width) + ") could not find a solution from state " + s.print());
                return std::make_pair(false, states_pruned);
            } catch (const std::exception& e) {
                spdlog::error("IW(" + std::to_string(_width) + ") failed solving from state " + s.print() + ". Reason: " + e.what());
                throw;
            }
        }

    private :
        Domain& _domain;
        unsigned int _width;
        std::function<void (const State&, const std::function<void (const unsigned int&, const bool&)>&)> _state_binarizer;
        std::function<bool (const State&)> _termination_checker;
        Graph& _graph;
        bool _debug_logs;
        ExecutionPolicy _execution_policy;

        unsigned int novelty(std::unordered_set<unsigned int>& true_bits, const State& s) const {
            unsigned int nov = 0;
            _state_binarizer(s, [&nov, &true_bits](const unsigned int& i, const bool& b){
                if (b) {
                    nov += (int) true_bits.insert(i).second;
                }
            });
            return nov;
        }
    };
};

} // namespace airlaps

#endif // AIRLAPS_IW_HH
