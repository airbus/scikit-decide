/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_RIW_HH
#define AIRLAPS_RIW_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

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


/** Use Environment domain knowledge for rollouts */
template <typename Tdomain>
struct EnvironmentRollout {
    std::list<typename Tdomain::Event> action_prefix;

    void init_rollout(Tdomain& domain) {
        domain.reset();
        std::for_each(action_prefix.begin(), action_prefix.end(),
                      [&domain](const typename Tdomain::Event& a){domain.step(a);});
    }

    std::unique_ptr<typename Tdomain::TransitionOutcome> progress(Tdomain& domain,
                                                                  const typename Tdomain::State& state,
                                                                  const typename Tdomain::Event& action) {
        return domain.step(action);
    }

    void advance(Tdomain& domain,
                 const typename Tdomain::State& state,
                 const typename Tdomain::Event& action,
                 bool record_action) {
        if (record_action) {
            action_prefix.push_back(action);
        } else {
            domain.step(action);
        }
    }
};


/** Use Simulation domain knowledge for rollouts */
template <typename Tdomain>
struct SimulationRollout {
    void init_rollout(Tdomain& domain) {}

    std::unique_ptr<typename Tdomain::TransitionOutcome> progress(Tdomain& domain,
                                                                  const typename Tdomain::State& state,
                                                                  const typename Tdomain::Event& action) {
        return domain.sample(state, action);
    }

    void advance(Tdomain& domain,
                 const typename Tdomain::State& state,
                 const typename Tdomain::Event& action,
                 bool record_action) {}
};


template <typename Tdomain,
          typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          template <typename...> class Trollout_policy = EnvironmentRollout,
          typename Texecution_policy = ParallelExecution>
class RIWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    typedef Trollout_policy<Domain> RolloutPolicy;
    typedef Texecution_policy ExecutionPolicy;

    RIWSolver(Domain& domain,
              const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
              unsigned int time_budget = 3600000,
              unsigned int rollout_budget = 100000,
              unsigned int max_depth = 1000,
              double exploration = 0.25,
              bool debug_logs = false)
    : _domain(domain), _state_features(state_features),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _exploration(exploration),
      _min_cost(std::numeric_limits<double>::max()), _max_cost(-std::numeric_limits<double>::max()),
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
        _min_cost = std::numeric_limits<double>::max();
        _max_cost = -std::numeric_limits<double>::max();
    }

    // solves from state s
    void solve(const State& s) {
        try {
            spdlog::info("Running " + ExecutionPolicy::print() + " RIW solver from state " + s.print());
            auto start_time = std::chrono::high_resolution_clock::now();
            unsigned int nb_rollouts = 0;
            unsigned int nb_of_binary_features = _state_features(s)->size();

            TupleVector feature_tuples;

            for (unsigned int w = 1 ; w <= nb_of_binary_features ; w++) {
                if(WidthSolver(*this, _domain, _state_features,
                               _time_budget, _rollout_budget,
                               _max_depth, _exploration,
                               _min_cost, _max_cost,
                               w, _graph, _rollout_policy,
                               _debug_logs).solve(s, start_time, nb_rollouts, feature_tuples)) {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                    spdlog::info("RIW finished to solve from state " + s.print() +
                                 " in " + std::to_string((double) duration / (double) 1e9) + " seconds with " +
                                 std::to_string(nb_rollouts) + " rollouts.");
                    return;
                }
            }

            spdlog::info("RIW could not find a solution from state " + s.print());
        } catch (const std::exception& e) {
            spdlog::error("RIW failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    bool is_solution_defined_for(const State& s) const {
        auto si = _graph.find(Node(s));
        if ((si == _graph.end()) || (si->best_action == nullptr) || (si->solved == false)) {
            return false;
        } else {
            return true;
        }
    }

    Action get_best_action(const State& s) {
        auto si = _graph.find(Node(s));
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        _rollout_policy.advance(_domain, s, *(si->best_action), true);
        Action best_action = *(si->best_action);
        std::unordered_set<Node*> root_subgraph;
        compute_reachable_subgraph(const_cast<Node&>(*si), root_subgraph); // we won't change the real key (Node::state) so we are safe
        Node* next_node = nullptr;
        for (auto& child : si->children) {
            if (&std::get<0>(child) == si->best_action) {
                next_node = std::get<2>(child);
                break;
            }
        }
        assert(next_node != nullptr);
        std::unordered_set<Node*> child_subgraph;
        compute_reachable_subgraph(*next_node, child_subgraph);
        update_graph(root_subgraph, child_subgraph);
        return best_action;
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(Node(s));
        if (si == _graph.end()) {
            spdlog::error("AIRLAPS exception: no best action found in state " + s.print());
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->fscore;
    }

private :

    Domain& _domain;
    std::function<std::unique_ptr<FeatureVector> (const State& s)> _state_features;
    unsigned int _time_budget;
    unsigned int _rollout_budget;
    unsigned int _max_depth;
    double _exploration;
    double _min_cost;
    double _max_cost;
    RolloutPolicy _rollout_policy;
    bool _debug_logs;

    struct Node {
        State state;
        std::unique_ptr<FeatureVector> features;
        std::vector<std::tuple<Action, double, Node*>> children;
        std::unordered_set<Node*> parents;
        double fscore; // not in A*'s meaning but rather to store cost-to-go once a solution is found
        unsigned int depth;
        unsigned int novelty;
        Action* best_action;
        bool terminal; // true if seen terminal from the simulator's perspective
        bool goal; // true if goal
        bool solved; // from this node: true if all reached states are either max_depth, or terminal or goal

        Node(const State& s, const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features)
            : state(s),
              fscore(std::numeric_limits<double>::infinity()),
              depth(std::numeric_limits<unsigned int>::max()),
              novelty(std::numeric_limits<unsigned int>::max()),
              best_action(nullptr),
              terminal(false),
              goal(false),
              solved(false) {
            features = state_features(s);
        }

        // Following constructor used only to search for the same existing node in the graph since nodes
        // are hashed by their states. Don't use this constructor for creating valid nodes!
        Node(const State& s) : state(s) {}
        
        struct Key {
            const typename HashingPolicy::Key& operator()(const Node& n) const {
                return HashingPolicy::get_key(n);
            }
        };
    };

    typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
    Graph _graph;

    typedef std::vector<std::pair<unsigned int, typename FeatureVector::value_type>> TupleType; // pair of var id and var value
    typedef std::vector<std::unordered_map<TupleType, unsigned int, boost::hash<TupleType>>> TupleVector; // mapped to min reached depth

    class WidthSolver { // known as IW(i), i.e. the fixed-width solver sequentially run by IW
    public :
        typedef Tdomain Domain;
        typedef typename Domain::State State;
        typedef typename Domain::Event Action;
        typedef Tfeature_vector FeatureVector;
        typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
        typedef Trollout_policy<Domain> RolloutPolicy;
        typedef Texecution_policy ExecutionPolicy;

        WidthSolver(RIWSolver& parent_solver, Domain& domain,
                    const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
                    unsigned int time_budget,
                    unsigned int rollout_budget,
                    unsigned int max_depth,
                    double exploration,
                    double& min_cost,
                    double& max_cost,
                    unsigned int width,
                    Graph& graph,
                    RolloutPolicy& rollout_policy,
                    bool debug_logs)
            : _parent_solver(parent_solver), _domain(domain), _state_features(state_features),
              _time_budget(time_budget), _rollout_budget(rollout_budget),
              _max_depth(max_depth), _exploration(exploration),
              _min_cost(min_cost), _max_cost(max_cost),
              _width(width), _graph(graph), _rollout_policy(rollout_policy),
              _debug_logs(debug_logs) {}
        
        // solves from state s
        // return true iff no state has been pruned or time or rollout budgets are consumed
        bool solve(const State& s,
                   const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time,
                   unsigned int& nb_rollouts,
                   TupleVector& feature_tuples) {
            try {
                spdlog::info("Running " + ExecutionPolicy::print() + " RIW(" + std::to_string(_width) + ") solver from state " + s.print());
                auto local_start_time = std::chrono::high_resolution_clock::now();
                std::random_device rd;
                std::mt19937 gen(rd());

                // Clear the solved bits
                // /!\ 'solved' bit set to 1 in RIW even if no solution found with previous width so we need to clear all the bits
                std::for_each(_graph.begin(), _graph.end(), [](const Node& n){
                    const_cast<Node&>(n).solved = false; // we don't change the real key (Node::state) so we are safe
                });

                // Create the root node containing the given state s
                auto si = _graph.emplace(Node(s, _state_features));
                Node& root_node = const_cast<Node&>(*(si.first)); // we won't change the real key (Node::state) so we are safe
                root_node.depth = 0;
                bool states_pruned = false;

                // Vector of sets of state feature tuples generated so far, for each w <= _width
                if (feature_tuples.size() < _width) {
                    feature_tuples.push_back(typename TupleVector::value_type());
                }
                novelty(feature_tuples, root_node, true); // initialize feature_tuples with the root node's bits

                // Start rollouts
                while (!root_node.solved &&
                       std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < _time_budget &&
                       nb_rollouts < _rollout_budget) {

                    // Start new rollout
                    nb_rollouts += 1;
                    Node* current_node = &root_node;

                    if (_debug_logs) spdlog::debug("New rollout from state: " + current_node->state.print() +
                                                   ", depth=" + std::to_string(current_node->depth) +
                                                   ", fscore=" + std::to_string(current_node->fscore));
                    _rollout_policy.init_rollout(_domain);

                    while (!(current_node->solved)) {
                        
                        if (_debug_logs) spdlog::debug("Current state: " + current_node->state.print() +
                                                       ", depth=" + std::to_string(current_node->depth) +
                                                       ", fscore=" + std::to_string(current_node->fscore));

                        if (current_node->children.empty()) {
                            // Generate applicable actions
                            auto applicable_actions = _domain.get_applicable_actions(current_node->state)->get_elements();
                            std::for_each(applicable_actions.begin(), applicable_actions.end(), [this, &current_node](const auto& a){
                                current_node->children.push_back(std::make_tuple(a, 0, nullptr));
                            });
                        }
                        
                        // Sample unsolved child
                        std::vector<unsigned int> unsolved_children;
                        std::vector<double> probabilities;
                        for (unsigned int i = 0 ; i < current_node->children.size() ; i++) {
                            Node* n = std::get<2>(current_node->children[i]);
                            if (!n) {
                                unsolved_children.push_back(i);
                                probabilities.push_back(_exploration);
                            } else if (!n->solved) {
                                unsolved_children.push_back(i);
                                probabilities.push_back((1.0 - _exploration) / ((double) n->novelty));
                            }
                        }
                        unsigned int pick = unsolved_children[std::discrete_distribution<>(probabilities.begin(), probabilities.end())(gen)];
                        bool new_node = false;

                        if (fill_child_node(current_node, pick, new_node)) { // terminal state
                            if (_domain.is_goal(current_node->state)) { // goal state
                                current_node->goal = true;
                                if (_debug_logs) spdlog::debug("Found a goal state: " + current_node->state.print() +
                                                            ", depth=" + std::to_string(current_node->depth) +
                                                            ", fscore=" + std::to_string(current_node->fscore));
                            } else { // dead-end state
                                if (_debug_logs) spdlog::debug("Found a dead-end state: " + current_node->state.print() +
                                                            ", depth=" + std::to_string(current_node->depth) +
                                                            ", fscore=" + std::to_string(current_node->fscore));
                            }
                            update_node(*current_node, true);
                            break;
                        } else if (!novelty(feature_tuples, *current_node, new_node)) { // no new tuple or not reached with lower depth => terminal node
                            if (_debug_logs) spdlog::debug("Pruning state: " + current_node->state.print() +
                                                           ", depth=" + std::to_string(current_node->depth) +
                                                           ", fscore=" + std::to_string(current_node->fscore));
                            states_pruned = true;
                            // /!\ current_node can become solved with some unsolved children in case it was
                            // already visited and novel but now some of its features are reached with lower depth
                            update_node(*current_node, true);
                            break;
                        } else if (current_node->depth >= _max_depth) {
                            if (_debug_logs) spdlog::debug("Max depth reached in state: " + current_node->state.print() +
                                                           ", depth=" + std::to_string(current_node->depth) +
                                                           ", fscore=" + std::to_string(current_node->fscore));
                            update_node(*current_node, true);
                            break;
                        } else if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() >= _time_budget) {
                            if (_debug_logs) spdlog::debug("Time budget consumed in state: " + current_node->state.print() +
                                                           ", depth=" + std::to_string(current_node->depth) +
                                                           ", fscore=" + std::to_string(current_node->fscore));
                            // next test: unexpanded node considered as a temporary (i.e. not solved) terminal node
                            // don't backup expanded node at this point otherwise the fscore initialization in update_node is wrong!
                            if (current_node->children.empty()) {
                                update_node(*current_node, false);
                            }
                            break;
                        }
                    }
                }

                if (_debug_logs) spdlog::debug("time budget: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()) +
                                               " ms, rollout budget: " + std::to_string(nb_rollouts) +
                                               ", states pruned: " + std::to_string(states_pruned));

                if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < _time_budget &&
                    nb_rollouts < _rollout_budget &&
                    states_pruned) {
                    spdlog::info("RIW(" + std::to_string(_width) + ") could not find a solution from state " + s.print());
                    return false;
                } else {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - local_start_time).count();
                    spdlog::info("RIW(" + std::to_string(_width) + ") finished to solve from state " + s.print() + " in " + std::to_string((double) duration / (double) 1e9) + " seconds.");
                    return true;
                }
            } catch (const std::exception& e) {
                spdlog::error("RIW(" + std::to_string(_width) + ") failed solving from state " + s.print() + ". Reason: " + e.what());
                throw;
            }
        }

    private :
        RIWSolver& _parent_solver;
        Domain& _domain;
        const std::function<std::unique_ptr<FeatureVector> (const State& s)>& _state_features;
        unsigned int _time_budget;
        unsigned int _rollout_budget;
        unsigned int _max_depth;
        double _exploration;
        double& _min_cost;
        double& _max_cost;
        unsigned int _width;
        Graph& _graph;
        RolloutPolicy& _rollout_policy;
        ExecutionPolicy _execution_policy;
        bool _debug_logs;

        // Input: feature tuple vector, node for which to compute novelty depth, boolean indicating whether this node is new or not
        // Returns true if at least one tuple is new or is reached with lower depth
        bool novelty(TupleVector& feature_tuples, Node& n, bool nn) const {
            // feature_tuples is a set of state variable combinations of size _width
            unsigned int nov = n.features->size() + 1;
            const FeatureVector& state_features = *n.features;
            bool novel_depth = false;

            for (unsigned int k = 1 ; k <= std::min(_width, (unsigned int) state_features.size()) ; k++) {
                // we must recompute combinations from previous width values just in case
                // this state would be visited for the first time across width iterations
                generate_tuples(k, state_features.size(),
                                [this, &state_features, &feature_tuples, &k, &novel_depth, &n, &nn, &nov](TupleType& cv){
                    for (auto& e : cv) {
                        e.second = state_features[e.first];
                    }
                    auto it = feature_tuples[k-1].insert(std::make_pair(cv, n.depth));
                    novel_depth = novel_depth || it.second || (nn && (it.first->second > n.depth)) || (!nn && (it.first->second == n.depth));
                    it.first->second = std::min(it.first->second, n.depth);
                    if(it.second) {
                        nov = std::min(nov, k);
                    }
                });
            }
            n.novelty = nov;
            if (_debug_logs) spdlog::debug("Novelty: " + std::to_string(nov));
            if (_debug_logs) spdlog::debug("Novelty depth check: " + std::to_string(novel_depth));
            return novel_depth;
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

        // Get the state reachable by calling the simulator from given node by applying given action number
        // Sets given node to the next one and returns whether the next one is terminal or not
        bool fill_child_node(Node*& node, unsigned int action_number, bool& new_node) {
            if (_debug_logs) spdlog::debug("Applying action: " + std::get<0>(node->children[action_number]).print());
            if (!std::get<2>(node->children[action_number])) { // first visit
                // Sampled child has not been visited so far, so generate it
                auto outcome = _rollout_policy.progress(_domain, node->state, std::get<0>(node->children[action_number]));
                auto i = _graph.emplace(Node(outcome->state(), _state_features));
                new_node = i.second;
                std::get<2>(node->children[action_number]) = &const_cast<Node&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                std::get<1>(node->children[action_number]) = outcome->cost();
                std::get<2>(node->children[action_number])->parents.insert(node);
                _min_cost = std::min(_min_cost, outcome->cost());
                _max_cost = std::max(_max_cost, outcome->cost());
                if (new_node) {
                    if (_debug_logs) spdlog::debug("Exploring new outcome: " + i.first->state.print() +
                                                   ", depth=" + std::to_string(i.first->depth) +
                                                   ", fscore=" + std::to_string(i.first->fscore));
                    std::get<2>(node->children[action_number])->depth = node->depth + 1;
                    node = std::get<2>(node->children[action_number]);
                    node->terminal = outcome->terminal();
                } else { // outcome already explored
                    if (_debug_logs) spdlog::debug("Exploring known outcome: " + i.first->state.print() +
                                                   ", depth=" + std::to_string(i.first->depth) +
                                                   ", fscore=" + std::to_string(i.first->fscore));
                    std::get<2>(node->children[action_number])->depth = std::min(
                        std::get<2>(node->children[action_number])->depth, node->depth + 1);
                    if (std::get<2>(node->children[action_number])->solved) { // solved child
                        node = std::get<2>(node->children[action_number]);
                        return true; // consider solved node as terminal to stop current rollout
                    }
                }
            } else { // second visit, unsolved child
                new_node = false;
                // call the simulator to be coherent with the new current node /!\ Assumes deterministic environment!
                _rollout_policy.advance(_domain, node->state, std::get<0>(node->children[action_number]), false);
                std::get<2>(node->children[action_number])->depth = std::min(
                    std::get<2>(node->children[action_number])->depth, node->depth + 1);
                node = std::get<2>(node->children[action_number]);
                if (_debug_logs) spdlog::debug("Exploring known outcome: " + node->state.print() +
                                               ", depth=" + std::to_string(node->depth) +
                                               ", fscore=" + std::to_string(node->fscore));
            }
            return node->terminal;
        }

        void update_node(Node& node, bool solved) {
            node.solved = solved;
            node.fscore = 0;
            std::unordered_set<Node*> frontier;
            frontier.insert(&node);
            _parent_solver.backup_values(frontier);
        }
    }; // WidthSolver class

    void compute_reachable_subgraph(Node& node, std::unordered_set<Node*>& subgraph) {
        std::unordered_set<Node*> frontier;
        frontier.insert(&node);
        subgraph.insert(&node);
        while(!frontier.empty()) {
            std::unordered_set<Node*> new_frontier;
            for (auto& n : frontier) {
                if (n) {
                    for (auto& child : n->children) {
                        if (subgraph.find(std::get<2>(child)) == subgraph.end()) {
                            new_frontier.insert(std::get<2>(child));
                            subgraph.insert(std::get<2>(child));
                        }
                    }
                }
            }
            frontier = new_frontier;
        }
    }

    // Prune the nodes that are no more reachable from the root's chosen child and reduce the depth of the
    // nodes reachable from the child node by 1
    void update_graph(std::unordered_set<Node*>& root_subgraph, std::unordered_set<Node*>& child_subgraph) {
        std::unordered_set<Node*> removed_subgraph;
        // First pass: look for nodes in root_subgraph but not child_subgraph and remove
        // those nodes from their children's parents
        // Don't actually remove those nodes in the first pass otherwise some children to remove
        // won't exist anymore when looking for their parents
        std::unordered_set<Node*> frontier;
        for (auto& n : root_subgraph) {
            if (n) {
                if (child_subgraph.find(n) == child_subgraph.end()) {
                    for (auto& child : n->children) {
                        if (std::get<2>(child)) {
                            std::get<2>(child)->parents.erase(n);
                        }
                    }
                    removed_subgraph.insert(n);
                } else {
                    n->depth -= 1;
                    // if (n->solved) {
                    if (n->children.empty()) {
                        frontier.insert(n);
                    }
                }
            }
        }
        // Second pass: actually remove nodes in root_subgraph but not in child_subgraph
        for (auto& n : removed_subgraph) {
            _graph.erase(Node(n->state));
        }
        // Third pass: recompute fscores
        backup_values(frontier);
    }

    // Backup values from tip solved nodes to their parents in graph
    void backup_values(std::unordered_set<Node*>& frontier) {
        unsigned int depth = 0; // used to prevent infinite loop in case of cycles
        for (auto& n : frontier) {
            if (n->terminal) {
                if (n->goal) {
                    // Applies a factor of 0.1 to min cost in order to favor goals
                    n->fscore = (_max_depth - n->depth) * _min_cost * ((_min_cost > 0)?0.1:10.0);
                } else {
                    n->fscore = std::numeric_limits<double>::infinity();
                }
            } else {
                // Applies a factor of 10 to max cost in order to favor non-pruned trajectories
                n->fscore = (_max_depth - n->depth) * _max_cost * ((_max_cost > 0)?10.0:0.1);
            }
        }

        while (!frontier.empty() && depth <= _max_depth) {
            depth += 1;
            std::unordered_set<Node*> new_frontier;
            for (auto& n : frontier) {
                for (auto& p : n->parents) {
                    p->solved = true;
                    p->fscore = std::numeric_limits<double>::infinity();
                    p->best_action = nullptr;
                    for (auto& nn : p->children) {
                        p->solved = p->solved && std::get<2>(nn) && std::get<2>(nn)->solved;
                        if (std::get<2>(nn)) {
                            double tentative_fscore = std::get<1>(nn) + std::get<2>(nn)->fscore;
                            if (p->fscore > tentative_fscore) {
                                p->fscore = tentative_fscore;
                                p->best_action = &std::get<0>(nn);
                            }
                        }
                    }
                    new_frontier.insert(p);
                }
            }
            frontier = new_frontier;
        }
    }
}; // RIWSolver class

} // namespace airlaps

#endif // AIRLAPS_RIW_HH