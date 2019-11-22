/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_WRL_HH
#define AIRLAPS_WRL_HH

#include <boost/container_hash/hash.hpp>
#include <random>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"

namespace airlaps {

/** Use default hasher provided with domain's states */
template <typename Tdomain, typename Tfeature_vector>
struct DomainStateHash {
    typedef const typename Tdomain::State& Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n) {
        return n.state;
    }

    template <typename Tstate, template Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tstate& s, const Tfeature_functor& state_features) {
        return std::unique_ptr<Tfeature_vector>();
    }

    template <typename Tnode, template Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> get_features(Tnode& n, const Tfeature_functor& state_features) {
        n.features = state_features(n.state);
        return n.features;
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

    template <typename Tstate, template Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tstate& s, const Tfeature_functor& state_features) {
        return state_features(s);
    }

    template <typename Tnode, template Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> get_features(Tnode& n, const Tfeature_functor& state_features) {
        return n.features;
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
          template <typename...> class Thashing_policy = DomainStateHash>
class WRLSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;

    class WRLDomainFilter {
    public :
        typedef typename Domain::EnvironmentOutcome EnvironmentOutcome;

        WRLDomainFilter(Domain& domain,
                        const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
                        double pruning_probability = 0.95,
                        double pruning_weight = 1000.0,
                        unsigned int width_increase_resilience = 10,
                        bool debug_logs = false)
            : _domain(domain), _state_features(state_features),
              _pruning_probability(pruning_probability), _pruning_weight(pruning_weight),
              _width_increase_resilience(width_increase_resilience), _debug_logs(debug_logs) {}
        
        virtual void clear() =0;
        virtual std::unique_ptr<State> reset() =0;
        virtual std::unique_ptr<EnvironmentOutcome> sample(const State& state, const Action& action) =0;
            
    protected :
        Domain& _domain;
        std::function<std::unique_ptr<FeatureVector> (const State& s)> _state_features;
        unsigned int _current_width;
        double _pruning_probability;
        double _pruning_weight;
        unsigned int _width_increase_resilience;
        double _min_reward;
        std::random_device _rd;
        std::mt19937 _gen;
        bool _debug_logs;

        typedef std::vector<std::pair<unsigned int, typename FeatureVector::value_type>> TupleType;
        typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>> TupleVector;
        TupleVector _feature_tuples;

        unsigned int novelty(TupleVector& feature_tuples, const FeatureVector& state_features) const {
            unsigned int nov = state_features.size() + 1;

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

    class WRLUncachedDomainFilter : public WRLDomainFilter {
    public :
        WRLUncachedDomainFilter(Domain& domain,
                                const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
                                double pruning_probability = 0.95,
                                double pruning_weight = 1000.0,
                                unsigned int width_increase_resilience = 1000,
                                bool debug_logs = false)
            : WRLDomainFilter(domain, state_features, pruning_probability, pruning_weigth,
                              width_increase_resilience, debug_logs),
              _nb_pruned_expansions(0) {}
        
        virtual void clear() {
            _feature_tuples = TupleVector(1);
            _current_width = 1;
            _nb_pruned_expansions = 0;
            _min_reward = std::numeric_limits<double>::max();
            _gen(_rd());
        }

        virtual std::unique_ptr<State> reset() {
            // Get original reset
            std::unique_ptr<State> s = _domain.reset();

            if (_feature_tuples.empty()) {
                this->clear();
            }

            std::unique_ptr<FeatureVector> features = _state_features(s);

            if (_nb_pruned_expansions >= _width_increase_resilience) {
                _current_width = std::min(_current_width + 1, features->size());
                _feature_tuples = TupleVector(_current_width);
            }

            novelty(_feature_tuples, *features)); // force computation of novelty and update of the feature tuples table

            return s;
        }

        virtual std::unique_ptr<EnvironmentOutcome> sample(const State& state, const Action& action) {
            // Get original sample
            std::unique_ptr<EnvironmentOutcome> o = _domain.sample(state, action);

            // Compute outcome's novelty
            std::unique_ptr<FeatureVector> features = _state_features(o->state);
            unsigned int nov = novelty(_feature_tuples, *features));
            bool pruned = (nov == features->size() + 1);

            if (pruned) {
                _nb_pruned_expansions += 1;
            } else {
                _nb_pruned_expansions = 0;
            }

            std::bernoulli_distribution d(_pruning_probability);

            if (pruned && d(_gen)) { // prune next state
                return std::make_unique<EnvironmentOutcome>(o->state,
                                                            _pruning_weight * ((_min_reward < 0.0)?_min_reward:(-1.0)), // make pruned state not attractive
                                                            true, // make pruned state terminal
                                                            o->info);
            } else {
                return std::make_unique<EnvironmentOutcome>(o->state,
                                                            o->reward / ((double) (nov)), // make state more attractive is low novelty
                                                            o->termination,
                                                            o->info);
            }
        }

    private :
        _nb_pruned_expansions; // number of pruned expansions in a row
    };

    class WRLCachedDomainFilter : public WRLDomainFilter {
    public :
        WRLCachedDomainFilter(Domain& domain,
                              const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features,
                              double pruning_probability = 0.95,
                              double pruning_weight = 1000.0,
                              unsigned int width_increase_resilience = 10,
                              bool debug_logs = false)
            : WRLDomainFilter(domain, state_features, pruning_probability, pruning_weigth,
                              width_increase_resilience, debug_logs) {}
        
        virtual void clear() {
            _graph.clear();
            _feature_tuples = TupleVector(1);
            _current_width = 1;
            _min_reward = std::numeric_limits<double>::max();
            _gen(_rd());
        }

        virtual std::unique_ptr<State> reset() {
            // Get original reset
            std::unique_ptr<State> s = _domain.reset();

            // Get the node containing the initial state
            auto si = _graph.emplace(Node(*s, _state_features));
            Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe

            if (si.second) { // first call
                if (!_feature_tuples.empty()) {
                    spdlog::warning("You SHOULD NOT use WRL in caching mode with non-deterministic initial state!");
                } else (_feature_tuples.empty()) {
                    this->clear();
                }
                node.novelty = novelty(_feature_tuples, *HashingPolicy::get_features(node, _state_features));
            }

            if (node.pruned) { // need for increasing novelty
                _current_width = std::min(_current_width + 1, node.features.size());
                _feature_tuples = TupleVector(_current_width);
                for (auto& n : _graph) {
                    n.nb_visits = 0;
                    n.pruned = false;
                    n.novelty = std::numeric_limits<unsigned int>::max();
                }
                node.novelty = novelty(_feature_tuples, *HashingPolicy::get_features(node, _state_features));
            }

            return s;
        }
    
        virtual std::unique_ptr<EnvironmentOutcome> sample(const State& state, const Action& action) {
            // Get the node containing the given state s
            auto si = _graph.emplace(Node(state, _state_features));
            Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe
            node.nb_visits += 1;
            Node* next_node = nullptr;

            if (si.second || si.first->children.find(action) == si.first->children.end()) { // state is not yet visited or not with same action
                // Get original sample
                std::unique_ptr<EnvironmentOutcome> o = _domain.sample(state, action);
                
                // Get the node containing the next state
                auto ni = _graph.emplace(Node(o->state, _state_features));
                next_node = const_cast<Node*>(ni.first); // we won't change the real key (StateNode::state) so we are safe
                node.children[action] = std::make_tuple(next_node, o->reward, o->termination, o->info);
                _min_reward = std::min(_min_reward, o->reward);
            } else {
                next_node = std::get<0>(node.children[action]);
            }

            if (next_node->novelty == std::numeric_limits<unsigned int>::max()) { // next state is not yet visited or previous width increment
                // Compute outcome's novelty
                next_node->novelty = novelty(_feature_tuples, *HashingPolicy::get_features(*next_node, _state_features));
                next_node->pruned = (next_node->novelty == next_node->features.size() + 1);
            }

            std::bernoulli_distribution d(_pruning_probability);
            const std::tuple<Node*, double, bool, typename EnvironmentOutcome::Info>& outcome = *node.children[action];

            if ((!node.pruned) && (node.nb_visits >= _width_increase_resilience) {
                node.pruned = true;
                for (auto& child : node.children) {
                    node.pruned = node.pruned && std::get<0>(child.second)->pruned;
                }
            }

            if (next_node->pruned && d(_gen)) { // prune next state
                return std::make_unique<EnvironmentOutcome>(std::get<0>(outcome)->state,
                                                            _pruning_weight * ((_min_reward < 0.0)?_min_reward:(-1.0)), // make pruned state not attractive
                                                            true, // make pruned state terminal
                                                            std::get<3>(outcome));
            } else {
                return std::make_unique<EnvironmentOutcome>(std::get<0>(outcome)->state,
                                                            std::get<1>(outcome) / ((double) (next_node->novelty)), // make state more attractive is low novelty
                                                            std::get<2>(outcome),
                                                            std::get<3>(outcome));
            }
        }

    private :
        struct Node {
            State state;
            std::unique_ptr<FeatureVector> features;
            unsigned int novelty;
            bool pruned; // true if pruned by the novelty test
            unsigned int nb_visits; // number of times the node is visited for current width

            typedef typename MapTypeDeducer<Action, std::tuple<Node*, double>, bool, typename EnvironmentOutcome::Info>::Map Children;
            Children children;

            Node(const State& s, const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features)
                : state(s),
                  novelty(std::numeric_limits<unsigned int>::max()),
                  pruned(false),
                  nb_visits(0) {
                features = HashingPolicy::node_create(s, state_features);
            }
            
            struct Key {
                const typename HashingPolicy::Key& operator()(const Node& n) const {
                    return HashingPolicy::get_key(n);
                }
            };
        };

        typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
        Graph _graph;
    };
};

}

#endif // AIRLAPS_WRL_HH
