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

    template <typename Tstate, typename Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tstate& s, const Tfeature_functor& state_features) {
        return std::unique_ptr<Tfeature_vector>();
    }

    template <typename Tnode, typename Tfeature_functor>
    static const Tfeature_vector& get_features(Tnode& n, const Tfeature_functor& state_features) {
        if (!n.features) {
            n.features = state_features(n.state);
        }
        return *n.features;
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

    template <typename Tstate, typename Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tstate& s, const Tfeature_functor& state_features) {
        return state_features(s);
    }

    template <typename Tnode, typename Tfeature_functor>
    static const Tfeature_vector& get_features(Tnode& n, const Tfeature_functor& state_features) {
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
          typename Tunderlying_solver,
          typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash>
class WRLSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Tunderlying_solver UnderlyingSolver;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    template <typename... T> using HashingPolicyTemplate = Thashing_policy<T...>;

    class WRLDomainFilter {
    public :
        typedef typename Domain::EnvironmentOutcome EnvironmentOutcome;
        typedef WRLSolver<Tdomain, Tunderlying_solver, Tfeature_vector, Thashing_policy> Solver;

        WRLDomainFilter(std::unique_ptr<Domain> domain,
                        const std::function<std::unique_ptr<FeatureVector> (const State&)>& state_features,
                        double initial_pruning_probability = 0.999,
                        double temperature_increase_rate = 0.01,
                        unsigned int width_increase_resilience = 10,
                        unsigned int max_depth = 1000,
                        bool debug_logs = false)
            : _domain(std::move(domain)), _state_features(state_features),
              _initial_pruning_probability(initial_pruning_probability),
              _temperature_increase_rate(temperature_increase_rate),
              _width_increase_resilience(width_increase_resilience),
              _max_depth(max_depth), _current_temperature(1.0),
              _min_reward(std::numeric_limits<double>::max()), _current_depth(0),
              _debug_logs(debug_logs) {}
        
        virtual void clear() =0;
        virtual std::unique_ptr<State> reset() =0;
        virtual std::unique_ptr<EnvironmentOutcome> step(const Action& action) =0;
        virtual std::unique_ptr<EnvironmentOutcome> sample(const State& state, const Action& action) =0;
            
    protected :
        std::unique_ptr<Domain> _domain;
        const std::function<std::unique_ptr<FeatureVector> (const State&)>& _state_features;
        unsigned int _current_width;
        double _initial_pruning_probability;
        double _temperature_increase_rate;
        unsigned int _width_increase_resilience;
        unsigned int _max_depth;
        double _current_temperature;
        double _min_reward;
        unsigned int _current_depth;
        std::random_device _rd;
        std::mt19937 _gen;
        bool _debug_logs;

        typedef std::vector<std::pair<unsigned int, typename FeatureVector::value_type>> TupleType;
        typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>> TupleVector;
        TupleVector _feature_tuples;

        unsigned int novelty(TupleVector& feature_tuples, const FeatureVector& state_features) const {
            unsigned int nov = state_features.size() + 1;

            for (unsigned int k = 1 ; k <= std::min(_current_width, (unsigned int) state_features.size()) ; k++) {
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
        WRLUncachedDomainFilter(std::unique_ptr<Domain> domain,
                                const std::function<std::unique_ptr<FeatureVector> (const State&)>& state_features,
                                double initial_pruning_probability = 0.999,
                                double temperature_increase_rate = 0.01,
                                unsigned int width_increase_resilience = 1000,
                                unsigned int max_depth = 1000,
                                bool debug_logs = false)
            : WRLDomainFilter(std::move(domain), state_features, initial_pruning_probability,
                              temperature_increase_rate, width_increase_resilience,
                              max_depth, debug_logs),
              _nb_pruned_expansions(0) {}
        
        virtual void clear() {
            this->_feature_tuples = typename WRLDomainFilter::TupleVector(1);
            this->_current_width = 1;
            this->_nb_pruned_expansions = 0;
            this->_min_reward = std::numeric_limits<double>::max();
            this->_gen = std::mt19937(this->_rd());
        }

        virtual std::unique_ptr<State> reset() {
            // Get original reset
            std::unique_ptr<State> s = this->_domain->reset();
            this->_current_depth = 0;
            std::unique_ptr<FeatureVector> features = this->_state_features(*s);

            if (this->_feature_tuples.empty()) { // first call
                this->clear();
                this->_current_temperature = - (features->size()) / std::log(1.0 - this->_initial_pruning_probability);
            } else {
                this->_current_temperature *= (1.0 + this->_temperature_increase_rate);
            }

            if (this->_nb_pruned_expansions >= this->_width_increase_resilience) {
                this->_current_width = std::min(this->_current_width + 1, (unsigned int) features->size());
                this->_feature_tuples = typename WRLDomainFilter::TupleVector(this->_current_width);
            }

            this->novelty(this->_feature_tuples, *features); // force computation of novelty and update of the feature tuples table

            return s;
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> step(const Action& action) {
            // Get original sample
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = this->_domain->step(action);
            this->_current_depth++;
            return make_transition(action, o);
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> sample(const State& state, const Action& action) {
            // Get original sample
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = this->_domain->sample(state, action);
            using EO = typename WRLDomainFilter::EnvironmentOutcome;
            this->_current_depth = EO::get_depth(o->info());
            return make_transition(action, o);
        }

        std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> make_transition(const Action& action,
                                                                                      std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome>& outcome) {
            // Compute outcome's novelty
            std::unique_ptr<FeatureVector> features = this->_state_features(outcome->state());
            unsigned int nov = this->novelty(this->_feature_tuples, *features);
            this->_nb_pruned_expansions = int(nov == features->size() + 1);

            std::bernoulli_distribution d(1.0 - std::exp((1.0 - nov) / this->_current_temperature));

            if (d(this->_gen)) { //prune next state
                outcome->reward(nov * (this->_max_depth - this->_current_depth) * this->_min_reward); // make attractiveness decreases with higher novelties
                outcome->termination(true); // make pruned state terminal
            }

            return std::move(outcome);
        }

    private :
        unsigned int _nb_pruned_expansions; // number of pruned expansions in a row
    };

    class WRLCachedDomainFilter : public WRLDomainFilter {
    public :
        WRLCachedDomainFilter(std::unique_ptr<Domain> domain,
                              const std::function<std::unique_ptr<FeatureVector> (const State&)>& state_features,
                              double initial_pruning_probability = 0.999,
                              double temperature_increase_rate = 0.01,
                              unsigned int width_increase_resilience = 10,
                              unsigned int max_depth = 1000,
                              bool debug_logs = false)
            : WRLDomainFilter(std::move(domain), state_features, initial_pruning_probability,
                              temperature_increase_rate, width_increase_resilience,
                              max_depth, debug_logs) {}
        
        virtual void clear() {
            this->_graph.clear();
            this->_feature_tuples = typename WRLDomainFilter::TupleVector(1);
            this->_current_width = 1;
            this->_current_state = State();
            this->_min_reward = std::numeric_limits<double>::max();
            this->_gen = std::mt19937(this->_rd());
        }

        virtual std::unique_ptr<State> reset() {
            // Get original reset
            std::unique_ptr<State> s = this->_domain->reset();
            this->_current_state = *s;
            this->_current_depth = 0;

            // Get the node containing the initial state
            auto si = this->_graph.emplace(Node(*s, this->_state_features));
            Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe
            node.depth = 0;

            if (si.second) { // first call
                if (!(this->_feature_tuples.empty())) {
                    spdlog::warn("You SHOULD NOT use WRL in caching mode with non-deterministic initial state!");
                } else {
                    this->clear();
                }
                const FeatureVector& features = HashingPolicy::get_features(node, this->_state_features);
                node.novelty = this->novelty(this->_feature_tuples, features);
                this->_current_temperature = - (features.size()) / std::log(1.0 - this->_initial_pruning_probability);
            } else {
                this->_current_temperature *= (1.0 + this->_temperature_increase_rate);
            }

            if (node.pruned) { // need for increasing width
                const FeatureVector& features = HashingPolicy::get_features(node, this->_state_features);
                this->_current_width = std::min(this->_current_width + 1, (unsigned int) features.size());
                this->_feature_tuples = typename WRLDomainFilter::TupleVector(this->_current_width);
                for (auto& n : _graph) {
                    const_cast<Node&>(n).nb_visits = 0; // we don't change the real key (Node::state) so we are safe
                    const_cast<Node&>(n).pruned = false; // we don't change the real key (Node::state) so we are safe
                    const_cast<Node&>(n).novelty = std::numeric_limits<unsigned int>::max(); // we don't change the real key (Node::state) so we are safe
                }
                node.novelty = this->novelty(this->_feature_tuples, features);
            }

            return s;
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> step(const Action& action) {
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = make_transition(
                this->_current_state, action,
                [this, &action](){
                    return this->_domain->step(action);
                });
            this->_current_state = o->state();
            return o;
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> sample(const State& state, const Action& action) {
             this->_current_state = state;
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = make_transition(
                state, action,
                [this, &state, &action](){
                    std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> eo = this->_domain->sample(state, action);
                    return eo;
                });
            this->_current_state = o->state();
            return o;
        }
    
        std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> make_transition(
            const State& state, const Action& action,
            const std::function<std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> ()>& transition) {
            // Get the node containing the given state s
            auto si = this->_graph.emplace(Node(state, this->_state_features));
            Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe
            node.nb_visits += 1;
            this->_current_depth = node.depth;
            Node* next_node = nullptr;

            if (si.second || si.first->children.find(action) == si.first->children.end()) { // state is not yet visited or not with same action
                // Get original sample
                std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = transition();
                
                // Get the node containing the next state
                auto ni = this->_graph.emplace(Node(o->state(), this->_state_features));
                next_node = &const_cast<Node&>(*ni.first); // we won't change the real key (StateNode::state) so we are safe
                node.children[action] = std::make_pair(next_node, std::move(o));
                this->_min_reward = std::min(this->_min_reward, o->reward());
                next_node->depth = std::min(next_node->depth, this->_current_depth + 1);
            } else {
                next_node = std::get<0>(node.children[action]);
            }

            if (next_node->novelty == std::numeric_limits<unsigned int>::max()) { // next state is not yet visited or previous width increment
                // Compute outcome's novelty
                next_node->novelty = this->novelty(this->_feature_tuples, HashingPolicy::get_features(*next_node, this->_state_features));
                next_node->pruned = (next_node->novelty == next_node->features->size() + 1);
            }

            std::bernoulli_distribution d(1.0 - std::exp((1.0 - (next_node->novelty)) / this->_current_temperature));
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> outcome = std::make_unique<typename WRLDomainFilter::EnvironmentOutcome>(*node.children[action].second);

            if ((!node.pruned) && (node.nb_visits >= this->_width_increase_resilience)) {
                node.pruned = true;
                for (auto& child : node.children) {
                    node.pruned = node.pruned && std::get<0>(child.second)->pruned;
                }
            }

            if (d(this->_gen)) { // prune next state
                outcome->reward((next_node->novelty) * (this->_max_depth - (next_node->depth)) * this->_min_reward); // make attractiveness decreases with higher novelties
                outcome->termination(true); // make pruned state terminal
            }

            return outcome;
        }

    private :
        struct Node {
            State state;
            std::unique_ptr<FeatureVector> features;
            unsigned int novelty;
            unsigned int depth;
            bool pruned; // true if pruned by the novelty test
            unsigned int nb_visits; // number of times the node is visited for current width

            typedef typename MapTypeDeducer<Action, std::pair<Node*, std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome>>>::Map Children;
            Children children;

            Node(const State& s, const std::function<std::unique_ptr<FeatureVector> (const State& s)>& state_features)
                : state(s),
                  novelty(std::numeric_limits<unsigned int>::max()),
                  depth(std::numeric_limits<unsigned int>::max()),
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
        State _current_state;
    };

    WRLSolver(UnderlyingSolver& solver,
              const std::function<std::unique_ptr<FeatureVector> (const State&)>& state_features,
              double initial_pruning_probability = 0.999,
              double temperature_increase_rate = 0.01,
              unsigned int width_increase_resilience = 10,
              unsigned int max_depth = 1000,
              bool cache_transitions = false,
              bool debug_logs = false)
        : _solver(solver), _state_features(state_features),
          _initial_pruning_probability(initial_pruning_probability),
          _temperature_increase_rate(temperature_increase_rate),
          _width_increase_resilience(width_increase_resilience),
          _max_depth(max_depth), _cache_transitions(cache_transitions),
          _debug_logs(debug_logs) {
        if (debug_logs) {
            spdlog::set_level(spdlog::level::debug);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
    }

    // Reset the solver
    void reset() {
        _solver.reset();
    }

    // solves using a given domain factory
    void solve(const std::function<std::unique_ptr<Domain> ()>& domain_factory) {
        _solver.solve(
            [this, &domain_factory]() -> std::unique_ptr<WRLDomainFilter> {
                if (_cache_transitions) {
                    return std::make_unique<WRLCachedDomainFilter>(
                        domain_factory(),
                        _state_features, _initial_pruning_probability, _temperature_increase_rate,
                        _width_increase_resilience, _max_depth, _debug_logs
                    );
                } else {
                    return std::make_unique<WRLUncachedDomainFilter>(
                        domain_factory(),
                        _state_features, _initial_pruning_probability, _temperature_increase_rate,
                        _width_increase_resilience, _max_depth, _debug_logs
                    );
                }
            }
        );
    }
    
private :
    UnderlyingSolver& _solver;
    bool _cache_transitions;
    std::unique_ptr<WRLDomainFilter> _filtered_domain;
    std::function<std::unique_ptr<FeatureVector> (const State&)> _state_features;
    unsigned int _current_width;
    double _initial_pruning_probability;
    double _temperature_increase_rate;
    unsigned int _width_increase_resilience;
    unsigned int _max_depth;
    bool _debug_logs;
};

}

#endif // AIRLAPS_WRL_HH
