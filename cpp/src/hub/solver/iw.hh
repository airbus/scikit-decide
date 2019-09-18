#ifndef AIRLAPS_IW_HH
#define AIRLAPS_IW_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include <queue>
#include <list>
#include <chrono>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace airlaps {

/** Use default hasher provided with domain's states */
template <typename Tdomain>
struct DomainStateHash {
    typedef const typename Tdomain::State& Key;

    DomainStateHash(const unsigned int& nb_of_binary_features,
                    const std::function<void (const typename Tdomain::State&, const std::function<void (const unsigned int&)>&)>& state_binarizer)
    : _nb_of_binary_features(nb_of_binary_features),
      _state_binarizer(state_binarizer) {}

    template <typename Tnode>
    Key get_key(const Tnode& node) const {
        return node.state;
    }

    template <typename Tnode>
    const std::vector<unsigned int>& get_state_features(const Tnode& node) const {
        node.state_features.clear();
        _state_binarizer(node.state, [this, &node](const unsigned int& i){
            if (i >= _nb_of_binary_features) {
                throw std::out_of_range("AIRLAPS exception: feature index " + std::to_string(i) +
                                        " exceeds the declared number of binary features (" +
                                        std::to_string(_nb_of_binary_features) + ")");
            }
            node.state_features.push_back(i);
        });
        return node.state_features;
    }

    struct Hash {
        std::size_t operator()(const Key& k) const {
            return typename Tdomain::State::Hash()(k);
        }
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const {
            return typename Tdomain::State::Equal()(k1, k2);
        }
    };

    unsigned int _nb_of_binary_features;
    std::function<void (const typename Tdomain::State&, const std::function<void (const unsigned int&)>&)> _state_binarizer;
};


/** Use state binary feature vector to hash states */
template <typename Tdomain>
struct StateFeatureHash {
    typedef std::vector<bool> Key;

    StateFeatureHash(const unsigned int& nb_of_binary_features,
                     const std::function<void (const typename Tdomain::State&, const std::function<void (const unsigned int&)>&)>& state_binarizer)
    : _nb_of_binary_features(nb_of_binary_features),
      _state_binarizer(state_binarizer) {}

    template <typename Tnode>
    Key get_key(const Tnode& node) const {
        node.state_features.clear();
        _state_binarizer(node.state, [this, &node](const unsigned int& i){
            if (i >= _nb_of_binary_features) {
                throw std::out_of_range("AIRLAPS exception: feature index " + std::to_string(i) +
                                        " exceeds the declared number of binary features (" +
                                        std::to_string(_nb_of_binary_features) + ")");
            }
            node.state_features.push_back(i);
        });
        Key bv(_nb_of_binary_features, false);
        for (const auto& i : node.state_features) {
            bv[i] = true;
        }
        return bv;
    }

    template <typename Tnode>
    const std::vector<unsigned int>& get_state_features(const Tnode& node) const {
        return node.state_features;
    }

    struct Hash {
        std::size_t operator()(const Key& k) const {
            return std::hash<std::vector<bool>>()(k);
        }
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const {
            return k1 == k2;
        }
    };

    unsigned int _nb_of_binary_features;
    std::function<void (const  typename Tdomain::State&, const std::function<void (const unsigned int&)>&)> _state_binarizer;
};


template <typename Tdomain,
          template <typename T> class Thashing_policy = DomainStateHash,
          typename Texecution_policy = ParallelExecution>
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Thashing_policy<Domain> HashingPolicy;
    typedef Texecution_policy ExecutionPolicy;

    IWSolver(Domain& domain,
             const unsigned int& nb_of_binary_features,
             const std::function<void (const State&, const std::function<void (const unsigned int&)>&)>& state_binarizer,
             bool debug_logs = false)
    : _domain(domain), _debug_logs(debug_logs) {
        if (debug_logs) {
            spdlog::set_level(spdlog::level::debug);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        _hashing_policy = std::make_unique<HashingPolicy>(nb_of_binary_features, state_binarizer);
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

            for (unsigned int w = 1 ; w <= _hashing_policy->_nb_of_binary_features ; w++) {
                std::pair<bool, bool> res = WidthSolver(_domain, w, _graph, _debug_logs, *_hashing_policy).solve(s);
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
        auto si = _graph.find(Node(s, *_hashing_policy));
        if ((si == _graph.end()) || (si->best_action == nullptr) || (si->solved == false)) {
            return false;
        } else {
            return true;
        }
    }

    const Action& get_best_action(const State& s) const {
        auto si = _graph.find(Node(s, *_hashing_policy));
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return *(si->best_action);
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(Node(s, *_hashing_policy));
        if (si == _graph.end()) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->fscore;
    }

private :

    Domain& _domain;
    bool _debug_logs;
    std::unique_ptr<HashingPolicy> _hashing_policy;

    struct Node {
        State state;
        mutable std::vector<unsigned int> state_features;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore; // not in A*'s meaning but rather to store cost-to-go once a solution is found
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool solved; // set to true if on the solution path constructed backward from the goal state
        const HashingPolicy& hashing_policy;

        Node(const State& s, const HashingPolicy& hp)
            : state(s),
              gscore(std::numeric_limits<double>::infinity()),
              fscore(std::numeric_limits<double>::infinity()),
              best_action(nullptr),
              solved(false),
              hashing_policy(hp) {}
        
        struct Key {
            typename HashingPolicy::Key operator()(const Node& n) const {
                return n.hashing_policy.get_key(n);
            }
        };
    };

    struct NodeCompare {
        bool operator()(Node*& a, Node*& b) const {
            return (a->gscore) > (b->gscore); // smallest element appears at the top of the priority_queue => cost optimization
        }
    };

    typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
    Graph _graph;

    class WidthSolver { // known as IW(i), i.e. the fixed-width solver sequentially run by IW
    public :
        typedef Tdomain Domain;
        typedef typename Domain::State State;
        typedef typename Domain::Event Action;
        typedef Thashing_policy<Domain> HashingPolicy;
        typedef Texecution_policy ExecutionPolicy;

        WidthSolver(Domain& domain,
                    unsigned int width,
                    Graph& graph,
                    bool debug_logs,
                    const HashingPolicy& hashing_policy)
            : _domain(domain), _width(width), _graph(graph), _debug_logs(debug_logs),
              _hashing_policy(hashing_policy) {}
        
        // solves from state s
        // returned pair p: p.first==true iff solvable, p.second==true iff states have been pruned
        std::pair<bool, bool> solve(const State& s) {
            try {
                spdlog::info("Running " + ExecutionPolicy::print() + " IW(" + std::to_string(_width) + ") solver from state " + s.print());
                auto start_time = std::chrono::high_resolution_clock::now();

                // Create the root node containing the given state s
                auto si = _graph.emplace(Node(s, _hashing_policy));
                if (si.first->solved || _domain.is_goal(s)) { // problem already solved from this state (was present in _graph and already solved)
                    return std::make_pair(true, false);
                } else if (_domain.is_terminal(s)) { // dead-end state
                    return std::make_pair(false, false);
                }
                Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
                root_node.gscore = 0;
                bool states_pruned = false;

                // Priority queue used to sort non-goal unsolved tip nodes by increasing cost-to-go values (so-called OPEN container)
                std::priority_queue<Node*, std::vector<Node*>, NodeCompare> open_queue;
                open_queue.push(&root_node);

                // Set of states that have already been explored
                std::unordered_set<Node*> closed_set;

                // Vector of sets of combinations (tuples) of Boolean state features generated so far, for each w <= _width
                std::vector<std::unordered_set<std::vector<bool>>> feature_combinations(_width);
                novelty(feature_combinations, root_node); // initialize feature_combinations with the root node's bits

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

                    if (_domain.is_goal(best_tip_node->state) || best_tip_node->solved) {
                        if (_debug_logs) spdlog::debug("Found a goal state: " + best_tip_node->state.print());
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

                    closed_set.insert(best_tip_node);

                    if (_domain.is_terminal(best_tip_node->state)) { // dead-end state
                        if (_debug_logs) spdlog::debug("Found a dead-end state: " + best_tip_node->state.print());
                        continue;
                    }

                    // Expand best tip node
                    auto applicable_actions = _domain.get_applicable_actions(best_tip_node->state)->get_elements();
                    std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node](const auto& a){
                        if (_debug_logs) spdlog::debug("Current expanded action: " + Action(a).print());
                        // Asynchronously compute next state distribution
                        // Must be separated from next loop in case the domain is python so that it is in this case actually implemented as a pool of independent processes
                        _domain.compute_next_state(best_tip_node->state, a);
                    });
                    std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node, &open_queue, &closed_set, &feature_combinations, &states_pruned](const auto& a){
                        auto next_state = _domain.get_next_state(best_tip_node->state, a);
                        std::pair<typename Graph::iterator, bool> i;
                        _execution_policy.protect([this, &i, &next_state]{
                            i = _graph.emplace(Node(next_state, _hashing_policy));
                        });
                        Node& neighbor = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                        if (_debug_logs) spdlog::debug("Exploring next state: " + neighbor.state.print());

                        if (closed_set.find(&neighbor) != closed_set.end()) {
                            // Ignore the neighbor which is already evaluated
                            return;
                        }

                        double transition_cost = _domain.get_transition_value(best_tip_node->state, a, neighbor.state);
                        double tentative_gscore = best_tip_node->gscore + transition_cost;

                        if ((i.second) || (tentative_gscore < neighbor.gscore)) {
                            if (_debug_logs) spdlog::debug("New gscore: " + std::to_string(best_tip_node->gscore) + "+" +
                                                           std::to_string(transition_cost) + "=" + std::to_string(tentative_gscore));
                            neighbor.gscore = tentative_gscore;
                            neighbor.best_parent = std::make_tuple(best_tip_node, a, transition_cost);
                        }

                        _execution_policy.protect([this, &feature_combinations, &open_queue, &neighbor, &states_pruned]{
                            if (novelty(feature_combinations, neighbor) > _width) {
                                if (_debug_logs) spdlog::debug("Pruning state");
                                states_pruned = true;
                            } else {
                                if (_debug_logs) spdlog::debug("Adding state to open queue");
                                open_queue.push(&neighbor);
                            }
                        });
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
        Graph& _graph;
        bool _debug_logs;
        ExecutionPolicy _execution_policy;
        const HashingPolicy& _hashing_policy;

        unsigned int novelty(std::vector<std::unordered_set<std::vector<bool>>>& feature_combinations,
                             Node& n) const {
            // feature_combinations is a set of Boolean combinations of size _width
            unsigned int nov = _hashing_policy._nb_of_binary_features + 1;
            const std::vector<unsigned int>& state_features = _hashing_policy.get_state_features(n);

            for (unsigned int k = 1 ; k <= std::min(_width, (unsigned int) state_features.size()) ; k++) {
                // we must recompute combinations from previous width values just in case
                // this state would be visited for the first time across width iterations
                generate_combinations(k, state_features.size(),
                                      [this, &state_features, &feature_combinations, &k, &nov](const std::vector<unsigned int>& cv){
                    std::vector<bool> bv(_hashing_policy._nb_of_binary_features, false);
                        for (const auto e : cv) {
                            bv[state_features[e]] = true;
                        }
                    if(feature_combinations[k-1].insert(bv).second) {
                        nov = std::min(nov, k);
                    }
                });
            }
            if (_debug_logs) spdlog::debug("Novelty: " + std::to_string(nov));
            return nov;
        }

        // Generates all combinations of size k from [0 ... (n-1)]
        void generate_combinations(const unsigned int& k,
                                   const unsigned int& n,
                                   const std::function<void (const std::vector<unsigned int>&)>& f) const {
            std::vector<unsigned int> cv(k); // one combination (the first one)
            std::iota(cv.begin(), cv.end(), 0);
            f(cv);
            bool more_combinations = true;
            while (more_combinations) {
                more_combinations = false;
                // find the rightmost element that has not yet reached its highest possible value
                for (unsigned int i = k; i > 0; i--) {
                    if (cv[i-1] < n - k + i - 1) {
                        // once finding this element, we increment it by 1,
                        // and assign the lowest valid value to all subsequent elements
                        cv[i-1]++;
                        for (unsigned int j = i; j < k; j++) {
                            cv[j] = cv[j-1] + 1;
                        }
                        f(cv);
                        more_combinations = true;
                        break;
                    }
                }
            }
        }
    };
};

} // namespace airlaps

#endif // AIRLAPS_IW_HH
