/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_WRL_HH
#define SKDECIDE_WRL_HH

#include <boost/container_hash/hash.hpp>
#include <random>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"

namespace skdecide {

/** Use default hasher provided with domain's observations */
template <typename Tdomain, typename Tfeature_vector>
struct DomainObservationHash {
    typedef typename Tdomain::Observation Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n) {
        return n.observation;
    }

    template <typename Tobservation, typename Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tobservation& s, const Tfeature_functor& observation_features) {
        return std::unique_ptr<Tfeature_vector>();
    }

    template <typename Tnode, typename Tfeature_functor>
    static const Tfeature_vector& get_features(Tnode& n, const Tfeature_functor& observation_features) {
        if (!n.features) {
            n.features = observation_features(n.observation);
        }
        return *n.features;
    }

    struct Hash {
        std::size_t operator()(const Key& k) const {
            return typename Tdomain::Observation::Hash()(k);
        }
    };

    struct Equal {
        bool operator()(const Key& k1, const Key& k2) const {
            return typename Tdomain::Observation::Equal()(k1, k2);
        }
    };
};


/** Use observation binary feature vector to hash observations */
template <typename Tdomain, typename Tfeature_vector>
struct ObservationFeatureHash {
    typedef Tfeature_vector Key;

    template <typename Tnode>
    static const Key& get_key(const Tnode& n) {
        return *n.features;
    }

    template <typename Tobservation, typename Tfeature_functor>
    static std::unique_ptr<Tfeature_vector> node_create(Tobservation& s, const Tfeature_functor& observation_features) {
        return observation_features(s);
    }

    template <typename Tnode, typename Tfeature_functor>
    static const Tfeature_vector& get_features(Tnode& n, const Tfeature_functor& observation_features) {
        return *n.features;
    }

    struct Hash {
        std::size_t operator()(const Key& k) const {
            std::size_t seed = 0;
            for (std::size_t i = 0 ; i < k.size() ; i++) {
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
            for (std::size_t i = 0 ; i < size ; i++) {
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
          template <typename...> class Thashing_policy = DomainObservationHash>
class WRLSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::Observation Observation;
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
                        const std::function<std::unique_ptr<FeatureVector> (const Observation&)>& observation_features,
                        double initial_pruning_probability = 0.999,
                        double temperature_increase_rate = 0.01,
                        std::size_t width_increase_resilience = 10,
                        std::size_t max_depth = 1000,
                        bool debug_logs = false)
            : _domain(std::move(domain)), _observation_features(observation_features),
              _initial_pruning_probability(initial_pruning_probability),
              _temperature_increase_rate(temperature_increase_rate),
              _width_increase_resilience(width_increase_resilience),
              _max_depth(max_depth), _current_temperature(1.0),
              _min_reward(std::numeric_limits<double>::max()), _current_depth(0),
              _debug_logs(debug_logs) {
                    if (debug_logs) {
                        spdlog::set_level(spdlog::level::debug);
                    } else {
                        spdlog::set_level(spdlog::level::info);
                    }
              }
        
        virtual ~WRLDomainFilter() {}
        
        virtual void clear() =0;
        virtual std::unique_ptr<Observation> reset() =0;
        virtual std::unique_ptr<EnvironmentOutcome> step(const Action& action) =0;
        virtual std::unique_ptr<EnvironmentOutcome> sample(const Observation& observation, const Action& action) =0;
            
    protected :
        std::unique_ptr<Domain> _domain;
        std::function<std::unique_ptr<FeatureVector> (const Observation&)> _observation_features;
        std::size_t _current_width;
        double _initial_pruning_probability;
        double _temperature_increase_rate;
        std::size_t _width_increase_resilience;
        std::size_t _max_depth;
        double _current_temperature;
        double _min_reward;
        std::size_t _current_depth;
        std::random_device _rd;
        std::mt19937 _gen;
        bool _debug_logs;

        typedef std::vector<std::pair<std::size_t, typename FeatureVector::value_type>> TupleType; // pair of var id and var value

        // Generates all combinations of size k from [0 ... (n-1)]
        void generate_tuples(const std::size_t& k,
                             const std::size_t& n,
                             const std::function<void (TupleType&)>& f) const {
            TupleType cv(k); // one combination (the first one)
            for (std::size_t i = 0 ; i < k ; i++) {
                cv[i].first = i;
            }
            f(cv);
            bool more_combinations = true;
            while (more_combinations) {
                more_combinations = false;
                // find the rightmost element that has not yet reached its highest possible value
                for (std::size_t i = k; i > 0; i--) {
                    if (cv[i-1].first < n - k + i - 1) {
                        // once finding this element, we increment it by 1,
                        // and assign the lowest valid value to all subsequent elements
                        cv[i-1].first++;
                        for (std::size_t j = i; j < k; j++) {
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
                                const std::function<std::unique_ptr<FeatureVector> (const Observation&)>& observation_features,
                                double initial_pruning_probability = 0.999,
                                double temperature_increase_rate = 0.01,
                                std::size_t width_increase_resilience = 1000,
                                std::size_t max_depth = 1000,
                                bool debug_logs = false)
            : WRLDomainFilter(std::move(domain), observation_features, initial_pruning_probability,
                              temperature_increase_rate, width_increase_resilience,
                              max_depth, debug_logs),
              _nb_pruned_expansions(0) {}
        
        virtual ~WRLUncachedDomainFilter() {}
        
        virtual void clear() {
            this->_feature_tuples = TupleVector(1);
            this->_current_width = 1;
            this->_nb_pruned_expansions = 0;
            this->_min_reward = std::numeric_limits<double>::max();
            this->_gen = std::mt19937(this->_rd());
        }

        virtual std::unique_ptr<Observation> reset() {
            try {
                if (this->_debug_logs) { spdlog::debug("Resetting the uncached width-based domain proxy"); }
                // Get original reset
                std::unique_ptr<Observation> s = this->_domain->reset();
                this->_current_depth = 0;
                std::unique_ptr<FeatureVector> features = this->_observation_features(*s);

                if (this->_feature_tuples.empty()) { // first call
                    this->clear();
                    this->_current_temperature = - ((double) features->size()) / std::log(1.0 - this->_initial_pruning_probability);
                    if (this->_debug_logs) { spdlog::debug("Initializing the uncached width-based proxy domain (temperature: " +
                                                           StringConverter::from(this->_current_temperature) + ")"); }
                } else {
                    this->_current_temperature *= (1.0 + this->_temperature_increase_rate);
                    if (this->_debug_logs) { spdlog::debug("Resetting the uncached width-based proxy domain (temperature: " +
                                                           StringConverter::from(this->_current_temperature) + ")"); }
                }

                if (this->_nb_pruned_expansions >= this->_width_increase_resilience) {
                    this->_current_width = std::min(this->_current_width + 1, (std::size_t) features->size());
                    this->_feature_tuples = TupleVector(this->_current_width);
                    this->_nb_pruned_expansions = 0;
                    if (this->_debug_logs) { spdlog::debug("Increase the width to " +
                                                           StringConverter::from(this->_current_width)); }
                }

                this->novelty(this->_feature_tuples, *features); // force computation of novelty and update of the feature tuples table

                return s;
            } catch (const std::exception& e) {
                spdlog::error("The uncached width-based proxy domain reset failed. Reason: " + std::string(e.what()));
                throw;
            }
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> step(const Action& action) {
            if (this->_debug_logs) { spdlog::debug("Stepping with action " + action.print()); }
            // Get original sample
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = this->_domain->step(action);
            this->_current_depth++;
            return make_transition(action, o);
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> sample(const Observation& observation, const Action& action) {
            if (this->_debug_logs) { spdlog::debug("Sampling with observation " + observation.print() + " and action " + action.print()); }
            // Get original sample
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = this->_domain->sample(observation, action);
            using EO = typename WRLDomainFilter::EnvironmentOutcome;
            this->_current_depth = EO::get_depth(o->info());
            return make_transition(action, o);
        }

        std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> make_transition(const Action& action,
                                                                                      std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome>& outcome) {
            try {
                // Compute outcome's novelty
                std::unique_ptr<FeatureVector> features = this->_observation_features(outcome->observation());
                std::size_t nov = this->novelty(this->_feature_tuples, *features);
                this->_nb_pruned_expansions += int(nov == (features->size() + 1));

                std::bernoulli_distribution d(1.0 - (std::exp((1.0 - nov) / this->_current_temperature)));
                this->_min_reward = std::min(this->_min_reward, outcome->reward());

                if (d(this->_gen)) { //prune next observation
                    outcome->reward(nov * (this->_max_depth - this->_current_depth) * this->_min_reward); // make attractiveness decreases with higher novelties
                    outcome->termination(true); // make pruned observation terminal
                    if (this->_debug_logs) { spdlog::debug("Pruning observation " + Observation(outcome->observation()).print() +
                                                           " (reward: " + StringConverter::from(outcome->reward()) +
                                                           "; probability: " + StringConverter::from(d.p()) + ")"); }
                } else if (this->_debug_logs) { spdlog::debug("Keeping observation " + Observation(outcome->observation()).print() +
                                                              " (min_reward: " + StringConverter::from(this->_min_reward) +
                                                              "; probability: " + StringConverter::from(1.0 - d.p()) + ")"); }

                return std::move(outcome);
            } catch (const std::exception& e) {
                spdlog::error("The uncached width-based proxy domain transition failed. Reason: " + std::string(e.what()));
                throw;
            }
        }

    private :
        std::size_t _nb_pruned_expansions; // number of pruned expansions in a row

        typedef typename WRLDomainFilter::TupleType TupleType;
        typedef std::vector<std::unordered_map<TupleType, std::size_t, boost::hash<TupleType>>> TupleVector; // mapped to min reached depth
        TupleVector _feature_tuples;

        std::size_t novelty(TupleVector& feature_tuples, const FeatureVector& observation_features) const {
            // feature_tuples is a set of state variable combinations of size _width
            std::size_t nov = observation_features.size() + 1;

            for (std::size_t k = 1 ; k <= std::min(this->_current_width, (std::size_t) observation_features.size()) ; k++) {
                // we must recompute combinations from previous width values just in case
                // this state would be visited for the first time across width iterations
                this->generate_tuples(k, observation_features.size(),
                                      [this, &observation_features, &feature_tuples, &k, &nov](TupleType& cv){
                    for (auto& e : cv) {
                        e.second = observation_features[e.first];
                    }
                    auto it = feature_tuples[k-1].insert(std::make_pair(cv, this->_current_depth));
                    if(it.second || (this->_current_depth <= it.first->second)) {
                        nov = std::min(nov, k);
                    }
                    it.first->second = std::min(it.first->second, this->_current_depth);
                });
            }
            if (this->_debug_logs) spdlog::debug("Novelty: " + StringConverter::from(nov));
            return nov;
        }
    };

    class WRLCachedDomainFilter : public WRLDomainFilter {
    public :
        WRLCachedDomainFilter(std::unique_ptr<Domain> domain,
                              const std::function<std::unique_ptr<FeatureVector> (const Observation&)>& observation_features,
                              double initial_pruning_probability = 0.999,
                              double temperature_increase_rate = 0.01,
                              std::size_t width_increase_resilience = 10,
                              std::size_t max_depth = 1000,
                              bool debug_logs = false)
            : WRLDomainFilter(std::move(domain), observation_features, initial_pruning_probability,
                              temperature_increase_rate, width_increase_resilience,
                              max_depth, debug_logs) {}
        
        virtual ~WRLCachedDomainFilter() {}
        
        virtual void clear() {
            this->_graph.clear();
            this->_feature_tuples = TupleVector(1);
            this->_current_width = 1;
            this->_min_reward = std::numeric_limits<double>::max();
            this->_gen = std::mt19937(this->_rd());
        }

        virtual std::unique_ptr<Observation> reset() {
            try {
                // Get original reset
                std::unique_ptr<Observation> s = this->_domain->reset();
                this->_current_observation = *s;
                this->_current_depth = 0;

                // Get the node containing the initial observation
                auto si = this->_graph.emplace(Node(*s, this->_observation_features));
                Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (ObservationNode::observation) so we are safe
                node.depth = 0;

                if (si.second) { // first call
                    if (!(this->_feature_tuples.empty())) {
                        spdlog::warn("You SHOULD NOT use the width-based proxy domain in caching mode with non-deterministic initial observation!");
                    } else {
                        this->clear();
                    }
                    const FeatureVector& features = HashingPolicy::get_features(node, this->_observation_features);
                    node.novelty = this->novelty(this->_feature_tuples, features);
                    this->_current_temperature = - ((double) features.size()) / std::log(1.0 - this->_initial_pruning_probability);
                    if (this->_debug_logs) { spdlog::debug("Initializing the cached width-based proxy domain (temperature: " +
                                                           StringConverter::from(this->_current_temperature) + ")"); }
                } else {
                    this->_current_temperature *= (1.0 + this->_temperature_increase_rate);
                    if (this->_debug_logs) { spdlog::debug("Resetting the cached width-based proxy domain (temperature: " +
                                                           StringConverter::from(this->_current_temperature) + ")"); }
                }

                if (node.pruned) { // need for increasing width
                    const FeatureVector& features = HashingPolicy::get_features(node, this->_observation_features);
                    this->_current_width = std::min(this->_current_width + 1, (std::size_t) features.size());
                    this->_feature_tuples = TupleVector(this->_current_width);
                    for (auto& n : _graph) {
                        const_cast<Node&>(n).nb_visits = 0; // we don't change the real key (Node::observation) so we are safe
                        const_cast<Node&>(n).pruned = false; // we don't change the real key (Node::observation) so we are safe
                        const_cast<Node&>(n).novelty = std::numeric_limits<std::size_t>::max(); // we don't change the real key (Node::observation) so we are safe
                    }
                    node.novelty = this->novelty(this->_feature_tuples, features);
                    if (this->_debug_logs) { spdlog::debug("Increase the width to " +
                                                           StringConverter::from(this->_current_width)); }
                }

                return s;
            } catch (const std::exception& e) {
                spdlog::error("The cached width-based proxy domain reset failed. Reason: " + std::string(e.what()));
                throw;
            }
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> step(const Action& action) {
            if (this->_debug_logs) { spdlog::debug("Stepping with action " + action.print()); }
            bool step_called = false;
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = make_transition(
                this->_current_observation, action,
                [this, &action, &step_called](){
                    step_called = true;
                    return this->_domain->step(action);
                });
            this->_current_observation = o->observation();
            if (!step_called) {
                this->_domain->step(action); // we need to step so we are ready for the potential next step
            }
            return o;
        }

        virtual std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> sample(const Observation& observation, const Action& action) {
            if (this->_debug_logs) { spdlog::debug("Sampling with observation " + observation.print() + " and action " + action.print()); }
            this->_current_observation = observation;
            std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = make_transition(
                observation, action,
                [this, &observation, &action](){
                    return this->_domain->sample(observation, action);
                });
            this->_current_observation = o->observation();
            return o;
        }
    
        std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> make_transition(
            const Observation& observation, const Action& action,
            const std::function<std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> ()>& transition) {
            try {
                // Get the node containing the given observation s
                auto si = this->_graph.emplace(Node(observation, this->_observation_features));
                Node& node = const_cast<Node&>(*(si.first)); // we won't change the real key (ObservationNode::observation) so we are safe
                node.nb_visits += 1;
                this->_current_depth = node.depth;
                Node* next_node = nullptr;

                if (si.second || si.first->children.find(action) == si.first->children.end()) { // observation is not yet visited or not with same action
                    if (this->_debug_logs) { spdlog::debug("Visiting new transition (calling the simulator)"); }

                    // Get original sample
                    std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> o = transition();
                    
                    // Get the node containing the next observation
                    auto ni = this->_graph.emplace(Node(o->observation(), this->_observation_features));
                    next_node = &const_cast<Node&>(*ni.first); // we won't change the real key (ObservationNode::observation) so we are safe
                    this->_min_reward = std::min(this->_min_reward, o->reward());
                    node.children[action] = std::make_pair(next_node, std::move(o));
                    next_node->depth = std::min(next_node->depth, this->_current_depth + 1);
                } else {
                    if (this->_debug_logs) { spdlog::debug("Visiting known transition (using the cache)"); }
                    next_node = std::get<0>(node.children[action]);
                }

                if (next_node->novelty == std::numeric_limits<std::size_t>::max()) { // next observation is not yet visited or previous width increment
                    // Compute outcome's novelty
                    next_node->novelty = this->novelty(this->_feature_tuples, HashingPolicy::get_features(*next_node, this->_observation_features));
                    next_node->pruned = (next_node->novelty == next_node->features->size() + 1);
                }

                std::bernoulli_distribution d(1.0 - (std::exp((1.0 - (next_node->novelty)) / this->_current_temperature)));
                std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome> outcome = std::make_unique<typename WRLDomainFilter::EnvironmentOutcome>(*node.children[action].second);

                if ((!node.pruned) && (node.nb_visits >= this->_width_increase_resilience)) {
                    node.pruned = true;
                    for (auto& child : node.children) {
                        node.pruned = node.pruned && std::get<0>(child.second)->pruned;
                    }
                }

                if (d(this->_gen)) { // prune next observation
                    outcome->reward((next_node->novelty) * (this->_max_depth - (next_node->depth)) * this->_min_reward); // make attractiveness decreases with higher novelties
                    outcome->termination(true); // make pruned observation terminal
                    if (this->_debug_logs) { spdlog::debug("Pruning observation " + Observation(outcome->observation()).print() +
                                                           " (reward: " + StringConverter::from(outcome->reward()) +
                                                           "; probability: " + StringConverter::from(d.p()) + ")"); }
                } else if (this->_debug_logs) { spdlog::debug("Keeping observation " + Observation(outcome->observation()).print() +
                                                              " (min_reward: " + StringConverter::from(this->_min_reward) +
                                                              "; probability: " + StringConverter::from(1.0 - d.p()) + ")"); }

                return outcome;
            } catch (const std::exception& e) {
                spdlog::error("The cached width-based proxy domain transition failed. Reason: " + std::string(e.what()));
                throw;
            }
        }

    private :
        struct Node {
            Observation observation;
            std::unique_ptr<FeatureVector> features;
            std::size_t novelty;
            std::size_t depth;
            bool pruned; // true if pruned by the novelty test
            std::size_t nb_visits; // number of times the node is visited for current width

            typedef typename MapTypeDeducer<Action, std::pair<Node*, std::unique_ptr<typename WRLDomainFilter::EnvironmentOutcome>>>::Map Children;
            Children children;

            Node(const Observation& s, const std::function<std::unique_ptr<FeatureVector> (const Observation& s)>& observation_features)
                : observation(s),
                  novelty(std::numeric_limits<std::size_t>::max()),
                  depth(std::numeric_limits<std::size_t>::max()),
                  pruned(false),
                  nb_visits(0) {
                features = HashingPolicy::node_create(s, observation_features);
            }
            
            struct Key {
                const typename HashingPolicy::Key& operator()(const Node& n) const {
                    return HashingPolicy::get_key(n);
                }
            };
        };

        typedef typename WRLDomainFilter::TupleType TupleType;
        typedef std::vector<std::unordered_set<TupleType, boost::hash<TupleType>>> TupleVector;
        TupleVector _feature_tuples;

        std::size_t novelty(TupleVector& feature_tuples, const FeatureVector& observation_features) const {
            std::size_t nov = observation_features.size() + 1;

            for (std::size_t k = 1 ; k <= std::min(this->_current_width, (std::size_t) observation_features.size()) ; k++) {
                // we must recompute combinations from previous width values just in case
                // this observation would be visited for the first time across width iterations
                this->generate_tuples(k, observation_features.size(),
                                      [&observation_features, &feature_tuples, &k, &nov](TupleType& cv){
                    for (auto& e : cv) {
                        e.second = observation_features[e.first];
                    }
                    if(feature_tuples[k-1].insert(cv).second) {
                        nov = std::min(nov, k);
                    }
                });
            }
            if (this->_debug_logs) spdlog::debug("Novelty: " + StringConverter::from(nov));
            return nov;
        }

        typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
        Graph _graph;
        Observation _current_observation;
    };

    WRLSolver(UnderlyingSolver& solver,
              const std::function<std::unique_ptr<FeatureVector> (const Observation&)>& observation_features,
              double initial_pruning_probability = 0.999,
              double temperature_increase_rate = 0.01,
              std::size_t width_increase_resilience = 10,
              std::size_t max_depth = 1000,
              bool cache_transitions = false,
              bool debug_logs = false)
        : _solver(solver), _cache_transitions(cache_transitions),
          _observation_features(observation_features),
          _initial_pruning_probability(initial_pruning_probability),
          _temperature_increase_rate(temperature_increase_rate),
          _width_increase_resilience(width_increase_resilience),
          _max_depth(max_depth), _debug_logs(debug_logs) {
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
        try {
            spdlog::info("Starting the width-based proxy domain solver");
            auto start_time = std::chrono::high_resolution_clock::now();
            _solver.solve(
                [this, &domain_factory]() -> std::unique_ptr<WRLDomainFilter> {
                    if (_cache_transitions) {
                        return std::make_unique<WRLCachedDomainFilter>(
                            domain_factory(),
                            _observation_features, _initial_pruning_probability, _temperature_increase_rate,
                            _width_increase_resilience, _max_depth, _debug_logs
                        );
                    } else {
                        return std::make_unique<WRLUncachedDomainFilter>(
                            domain_factory(),
                            _observation_features, _initial_pruning_probability, _temperature_increase_rate,
                            _width_increase_resilience, _max_depth, _debug_logs
                        );
                    }
                }
            );
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            spdlog::info("The width-based proxy domain solver finished to solve in " + StringConverter::from((double) duration / (double) 1e9) + " seconds.");
        } catch (const std::exception& e) {
            spdlog::error("The width-based proxy domain solver failed. Reason: " + std::string(e.what()));
            throw;
        }
    }
    
private :
    UnderlyingSolver& _solver;
    bool _cache_transitions;
    std::unique_ptr<WRLDomainFilter> _filtered_domain;
    std::function<std::unique_ptr<FeatureVector> (const Observation&)> _observation_features;
    std::size_t _current_width;
    double _initial_pruning_probability;
    double _temperature_increase_rate;
    std::size_t _width_increase_resilience;
    std::size_t _max_depth;
    bool _debug_logs;
};

}

#endif // SKDECIDE_WRL_HH
