#ifndef AIRLAPS_IW_HH
#define AIRLAPS_IW_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include <queue>
#include <list>
#include <chrono>

#include <boost/container_hash/hash.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace airlaps {

/** Use default hasher provided with domain's states */
template <typename Tdomain, typename Tfeature_vector>
struct DomainStateHash {
    typedef const typename Tdomain::State& Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n) {
        return n.state;
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
};


/** Use state binary feature vector to hash states */
template <typename Tdomain, typename Tfeature_vector>
struct StateFeatureHash {
    typedef Tfeature_vector Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n) {
        return *n.features;
    }

    struct Hash {
        std::size_t operator()(const Key& k) const {
            std::size_t seed = 0;
            for (unsigned int i = 0 ; i < k.size() ; i++) {
                boost::hash_combine(seed, k[i]);
            }
            return seed;
        }
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const {
            std::size_t size = k1.size();
            if (size != k2.size()) {
                return false;
            }
            for (unsigned int i = 0 ; i < size ; i++) {
                if (!(k1[i] == k2[i])) {
                    return false;
                }
            }
            return true;
        }
    };
};


template <typename Tdomain,
          typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          typename Texecution_policy = ParallelExecution>
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    typedef Texecution_policy ExecutionPolicy;

    IWSolver(Domain& domain,
             const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
             const std::function<bool (const double&, const unsigned int&, const unsigned int&,
                                       const double&, const unsigned int&, const unsigned int&)>& node_ordering = nullptr,
             bool debug_logs = false)
    : _domain(domain), _state_features(state_features), _debug_logs(debug_logs) {
        if (!node_ordering) {
            _node_ordering = [](const double& a_gscore, const unsigned int& a_novelty, const unsigned int&  a_depth,
                                const double& b_gscore, const unsigned int& b_novelty, const unsigned int& b_depth) -> bool {
                                    return a_gscore > b_gscore;
                                };
        } else {
            _node_ordering = node_ordering;
        }
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
            unsigned int nb_of_binary_features = _state_features(s)->size();

            for (unsigned int w = 1 ; w <= nb_of_binary_features ; w++) {
                std::pair<bool, bool> res = WidthSolver(_domain, _state_features, _node_ordering, w, _graph, _debug_logs).solve(s);
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
        auto si = _graph.find(Node(s, _state_features));
        if ((si == _graph.end()) || (si->best_action == nullptr) || (si->solved == false)) {
            return false;
        } else {
            return true;
        }
    }

    const Action& get_best_action(const State& s) const {
        auto si = _graph.find(Node(s, _state_features));
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return *(si->best_action);
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(Node(s, _state_features));
        if (si == _graph.end()) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->fscore;
    }

private :

    Domain& _domain;
    std::function<std::unique_ptr<FeatureVector> (const State& s)> _state_features;
    std::function<bool (const double&, const unsigned int&, const unsigned int&,
                                       const double&, const unsigned int&, const unsigned int&)> _node_ordering;
    bool _debug_logs;

    struct Node {
        State state;
        std::unique_ptr<FeatureVector> features;
        std::tuple<Node*, Action, double> best_parent;
        double gscore;
        double fscore; // not in A*'s meaning but rather to store cost-to-go once a solution is found
        unsigned int novelty;
        unsigned int depth;
        Action* best_action; // computed only when constructing the solution path backward from the goal state
        bool solved; // set to true if on the solution path constructed backward from the goal state

        Node(const State& s, const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features)
            : state(s),
              gscore(std::numeric_limits<double>::infinity()),
              fscore(std::numeric_limits<double>::infinity()),
              novelty(std::numeric_limits<unsigned int>::max()),
              depth(std::numeric_limits<unsigned int>::max()),
              best_action(nullptr),
              solved(false) {
            features = state_features(s);
        }
        
        struct Key {
            const typename HashingPolicy::Key& operator()(const Node& n) const {
                return HashingPolicy::get_key(n);
            }
        };
    };

    typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
    Graph _graph;

    class WidthSolver { // known as IW(i), i.e. the fixed-width solver sequentially run by IW
    public :
        typedef Tdomain Domain;
        typedef typename Domain::State State;
        typedef typename Domain::Event Action;
        typedef Tfeature_vector FeatureVector;
        typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
        typedef Texecution_policy ExecutionPolicy;

        WidthSolver(Domain& domain,
                    const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
                    const std::function<bool (const double&, const unsigned int&, const unsigned int&,
                                              const double&, const unsigned int&, const unsigned int&)>& node_ordering,
                    unsigned int width,
                    Graph& graph,
                    bool debug_logs)
            : _domain(domain), _state_features(state_features), _node_ordering(node_ordering),
              _width(width), _graph(graph), _debug_logs(debug_logs) {}
        
        // solves from state s
        // returned pair p: p.first==true iff solvable, p.second==true iff states have been pruned
        std::pair<bool, bool> solve(const State& s) {
            try {
                spdlog::info("Running " + ExecutionPolicy::print() + " IW(" + std::to_string(_width) + ") solver from state " + s.print());
                auto start_time = std::chrono::high_resolution_clock::now();

                // Create the root node containing the given state s
                auto si = _graph.emplace(Node(s, _state_features));
                if (si.first->solved || _domain.is_goal(s)) { // problem already solved from this state (was present in _graph and already solved)
                    return std::make_pair(true, false);
                } else if (_domain.is_terminal(s)) { // dead-end state
                    return std::make_pair(false, false);
                }
                Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
                root_node.depth = 0;
                root_node.gscore = 0;
                bool states_pruned = false;

                auto node_ordering = [this](const auto& a, const auto& b) -> bool {
                    return _node_ordering(a->gscore, a->novelty, a->depth,
                                          b->gscore, b->novelty, b->depth);
                };

                // Priority queue used to sort non-goal unsolved tip nodes by increasing cost-to-go values (so-called OPEN container)
                std::priority_queue<Node*, std::vector<Node*>, decltype(node_ordering)> open_queue(node_ordering);
                open_queue.push(&root_node);

                // Set of states that have already been explored
                std::unordered_set<Node*> closed_set;

                // Vector of sets of state feature tuples generated so far, for each w <= _width
                TupleVector feature_tuples(_width);
                novelty(feature_tuples, root_node); // initialize feature_tuples with the root node's bits

                while (!open_queue.empty()) {
                    auto best_tip_node = open_queue.top();
                    open_queue.pop();

                    // Check that the best tip node has not already been closed before
                    // (since this implementation's open_queue does not check for element uniqueness,
                    // it can contain many copies of the same node pointer that could have been closed earlier)
                    if (closed_set.find(best_tip_node) != closed_set.end()) { // this implementation's open_queue can contain several
                        continue;
                    }

                    if (_debug_logs) spdlog::debug("Current best tip node: " + best_tip_node->state.print() +
                                                   ", gscore=" + std::to_string(best_tip_node->gscore));

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
                    std::for_each(ExecutionPolicy::policy, applicable_actions.begin(), applicable_actions.end(), [this, &best_tip_node, &open_queue, &closed_set, &feature_tuples, &states_pruned](const auto& a){
                        auto next_state = _domain.get_next_state(best_tip_node->state, a);
                        std::pair<typename Graph::iterator, bool> i;
                        _execution_policy.protect([this, &i, &next_state]{
                            i = _graph.emplace(Node(next_state, _state_features));
                        });
                        Node& neighbor = const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                        if (_debug_logs) spdlog::debug("Exploring next state: " + neighbor.state.print() +
                                                       " (among " + std::to_string(_graph.size()) + ")");

                        double transition_cost = _domain.get_transition_value(best_tip_node->state, a, neighbor.state);
                        double tentative_gscore = best_tip_node->gscore + transition_cost;
                        unsigned int tentative_depth = best_tip_node->depth + 1;

                        if ((i.second) || (tentative_gscore < neighbor.gscore)) {
                            if (_debug_logs) spdlog::debug("New gscore: " + std::to_string(best_tip_node->gscore) + "+" +
                                                           std::to_string(transition_cost) + "=" + std::to_string(tentative_gscore));
                            neighbor.gscore = tentative_gscore;
                            neighbor.best_parent = std::make_tuple(best_tip_node, a, transition_cost);
                        }

                        if ((i.second) || (tentative_depth < neighbor.depth)) {
                            if (_debug_logs) spdlog::debug("New depth: " + std::to_string(best_tip_node->depth) + "+" +
                                                           std::to_string(1) + "=" + std::to_string(tentative_depth));
                            neighbor.depth = tentative_depth;
                        }

                        _execution_policy.protect([this, &feature_tuples, &open_queue, &neighbor, &states_pruned]{
                            if (novelty(feature_tuples, neighbor) > _width) {
                                if (_debug_logs) spdlog::debug("Pruning state");
                                states_pruned = true;
                            } else {
                                if (_debug_logs) spdlog::debug("Adding state to open queue (among " + std::to_string(open_queue.size()) + ")");
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
        const std::function<std::unique_ptr<FeatureVector> (const State& s)>& _state_features;
        const std::function<bool (const double&, const unsigned int&, const unsigned int&,
                                  const double&, const unsigned int&, const unsigned int&)>& _node_ordering;
        unsigned int _width;
        Graph& _graph;
        bool _debug_logs;
        ExecutionPolicy _execution_policy;

        typedef std::vector<std::pair<unsigned int, typename FeatureVector::value_type>> TupleType;
        typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>> TupleVector;

        unsigned int novelty(TupleVector& feature_tuples, Node& n) const {
            // feature_tuples is a set of state variable combinations of size _width
            unsigned int nov = n.features->size() + 1;
            const FeatureVector& state_features = *n.features;

            for (unsigned int k = 1 ; k <= std::min(_width, (unsigned int) state_features.size()) ; k++) {
                // we must recompute combinations from previous width values just in case
                // this state would be visited for the first time across width iterations
                generate_tuples(k, state_features.size(),
                                [this, &state_features, &feature_tuples, &k, &nov](TupleType& cv){
                    for (auto& e : cv) {
                        e.second = state_features[e.first];
                    }
                    if(feature_tuples[k-1].insert(cv).second) {
                        nov = std::min(nov, k);
                    }
                });
            }
            if (_debug_logs) spdlog::debug("Novelty: " + std::to_string(nov));
            n.novelty = nov;
            return nov;
        }

        // Generates all combinations of size k from [0 ... (n-1)]
        void generate_tuples(const unsigned int& k,
                             const unsigned int& n,
                             const std::function<void (TupleType&)>& f) const {
            TupleType cv(k); // one combination (the first one)
            for (unsigned int i = 0 ; i < k ; i++) {
                cv[i].first = i;
            }
            f(cv);
            bool more_combinations = true;
            while (more_combinations) {
                more_combinations = false;
                // find the rightmost element that has not yet reached its highest possible value
                for (unsigned int i = k; i > 0; i--) {
                    if (cv[i-1].first < n - k + i - 1) {
                        // once finding this element, we increment it by 1,
                        // and assign the lowest valid value to all subsequent elements
                        cv[i-1].first++;
                        for (unsigned int j = i; j < k; j++) {
                            cv[j].first = cv[j-1].first + 1;
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
