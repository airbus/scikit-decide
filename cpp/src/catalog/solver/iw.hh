#ifndef AIRLAPS_IW_HH
#define AIRLAPS_IW_HH

#include <exception>
#include <unordered_map>
#include <deque>
#include <vector>
#include <queue>
#include <chrono>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/execution.hh"

namespace airlaps {

template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef typename Domain::FeatureVector FeatureVector;
    typedef Texecution_policy ExecutionPolicy;

    IWSolver(Domain& domain,
             const std::function<FeatureVector (const State&)>& state_to_feature_atoms,
             size_t frameskip = 15,
             int simulator_budget = 150000,
             double time_budget = std::numeric_limits<double>::infinity(),
             bool novelty_subtables = false,
             bool random_actions = false,
             size_t max_rep = 30,
             int nodes_threshold = 50000,
             int lookahead_caching = 2,
             double discount = 1.0,
             bool debug_logs = false)
        : domain_(domain), state_to_feature_atoms_(state_to_feature_atoms),
          frameskip_(frameskip), simulator_budget_(simulator_budget),
          num_tracked_atoms_(0), time_budget_(time_budget),
          novelty_subtables_(novelty_subtables), random_actions_(random_actions),
          max_rep_(max_rep), nodes_threshold_(nodes_threshold), lookahead_caching_(lookahead_caching),
          discount_(discount), debug_logs_(debug_logs), execution_node_(nullptr) {
        if (debug_logs) {
            spdlog::set_level(spdlog::level::debug);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
    }

    virtual ~IWSolver() {
        // properly clear episode_.node in case we come from a previous episode
        // that was not properly ended by calling episode_end()
        if( execution_node_ != nullptr ) {
            assert(execution_node_->parent_ != nullptr);
            assert(execution_node_->parent_->parent_ == nullptr);
            remove_tree(execution_node_->parent_);
            execution_node_ = nullptr;
        }
    }

    // reset the solver
    void reset() {
        num_tracked_atoms_ = 0;
        // properly clear episode_.node in case we come from a previous episode
        // that was not properly ended by calling episode_end()
        if( execution_node_ != nullptr ) {
            assert(execution_node_->parent_ != nullptr);
            assert(execution_node_->parent_->parent_ == nullptr);
            remove_tree(execution_node_->parent_);
            execution_node_ = nullptr;
        }
    }

    // Solves from state s
    void solve(const State& s) {
        try  {
            auto start_time = std::chrono::high_resolution_clock::now();
            reset_stats();
            
            if( (execution_node_ != nullptr) && (lookahead_caching_ == 1) ) {
                execution_node_->clear_cached_observations();
                assert(execution_node_->state_ == nullptr);
                assert(execution_node_->parent_ != nullptr);
                assert(execution_node_->parent_->state_ != nullptr);
            }

            if (num_tracked_atoms_ == 0) {
                num_tracked_atoms_ = state_to_feature_atoms_(s).size();
            }

            Node* root = execution_node_;
            execution_branch_.clear();
            std::deque<Action>& branch = execution_branch_;

            spdlog::info("Solving from state " + s.print());
            spdlog::info(std::string("input:")
                        + " #nodes=" + (root == nullptr ? "na" : std::to_string(root->num_nodes()))
                        + ", #tips=" + (root == nullptr ? "na" : std::to_string(root->num_tip_nodes()))
                        + ", height=" + (root == nullptr ? "na" : std::to_string(root->height_)));
            
            // novelty table and other vars
            std::unordered_map<int, FeatureVector > novelty_table_map;

            // construct root node
            if( root == nullptr ) {
                Node *root_parent = new Node(nullptr, Action(), -1);
                root = new Node(root_parent, Action(), 0);
                root->state_ = std::make_unique<State>(s);
            }

            // if root has some children, make sure it has all children
            if( root->num_children_ > 0 ) {
                assert(root->first_child_ != nullptr);
                std::unordered_set<Action, typename Action::Hash, typename Action::Equal> root_actions;
                for( Node *child = root->first_child_; child != nullptr; child = child->sibling_ )
                    root_actions.insert(child->action_);

                // complete children
                auto applicable_actions = domain_.get_applicable_actions(*(root->state_))->get_elements();
                for (auto i = applicable_actions.begin(); i != applicable_actions.end(); i++) {
                    if( root_actions.find(*i) == root_actions.end() )
                        root->expand(*i);
                }
            } else {
                // make sure this root node isn't marked as frame rep
                root->parent_->feature_atoms_.clear();
            }

            // normalize depths, reset rep counters, and recompute path rewards
            root->parent_->depth_ = -1;
            root->normalize_depth();
            root->reset_frame_rep_counters(this->frameskip_);
            root->recompute_path_rewards(root);

            // now that everything has been prepared, actually solve the problem
            // from the root node
            if( int(root->num_nodes()) < nodes_threshold_ ) {
                do_solve(start_time, root, novelty_table_map);
            }

            // if nothing was expanded, return random actions (it can only happen with small time budget)
            if( root->num_children_ == 0 ) {
                assert(root->first_child_ == nullptr);
                assert(time_budget_ != std::numeric_limits<double>::infinity());
                random_decision_ = true;
                branch.push_back(Action());
            } else {
                assert(root->first_child_ != nullptr);

                // backup values and calculate heights
                root->backup_values(discount_);
                root->calculate_height();
                root_height_ = root->height_;

                // print info about root node
                if (this->debug_logs_) {
                    std::string debug_msg = std::string("root:")
                                        + " solved=" + std::to_string(root->solved_)
                                        + ", value=" + std::to_string(root->value_)
                                        + ", imm-reward=" + std::to_string(root->reward_)
                                        + ", children=[";
                    for( Node *child = root->first_child_; child != nullptr; child = child->sibling_ )
                        debug_msg += std::to_string(child->qvalue(discount_)) + ":" + child->action_.print() + " ";
                    debug_msg += "]";
                    spdlog::debug(debug_msg);
                }
            }

            // compute branch
            if( root->value_ != 0 ) {
                root->best_branch(branch, discount_);
            } else {
                if( random_actions_ ) {
                    random_decision_ = true;
                    branch.push_back(this->random_zero_value_action(root, discount_));
                } else {
                    root->longest_zero_value_branch(discount_, branch);
                    assert(!branch.empty());
                }
            }

            // make sure states along branch exist (only needed when doing partial caching)
            this->generate_states_along_branch(root, branch);

            // print branch
            assert(!branch.empty());
            spdlog::debug(std::string("branch:")
                          + " value=" + std::to_string(root->value_)
                          + ", size=" + std::to_string(branch.size())
                          + ", actions:");

            // stop timer and print stats
            auto end_time = std::chrono::high_resolution_clock::now();
            total_time_ = double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / double(1e6);
            print_stats(*root, novelty_table_map);

            // select action to apply
            Action action = branch.front();
            spdlog::info("executable-action: " + action.print());

            // advance/destroy lookhead tree
            if( execution_node_ != nullptr ) {
                if( (lookahead_caching_ == 0) || (execution_node_->num_children_ == 0) ) {
                    remove_tree(execution_node_);
                    execution_node_ = nullptr;
                } else {
                    assert(execution_node_->parent_->state_ != nullptr);
                    execution_node_ = execution_node_->advance(action);
                }
            }

        } catch (const std::exception& e) {
            spdlog::error("AIRLAPS exception: failed solving from state " + s.print() + ". Reason: " + e.what());
            throw;
        }
    }

    const Action& get_best_action(const State& s) const {
        if (execution_branch_.empty()) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print() + " (empty branch)");
        }
        return execution_branch_.front();
    }

    const double& get_best_value(const State& s) const {
        if (execution_node_ == nullptr) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print() + " (null execution node)");
        }
        return execution_node_->value_;
    }

protected :
    Domain& domain_;
    std::function<FeatureVector (const State&)> state_to_feature_atoms_;
    
    const size_t frameskip_;
    int random_seed_;

    const int simulator_budget_;
    size_t num_tracked_atoms_;

    const double time_budget_;
    const bool novelty_subtables_;
    const bool random_actions_;
    const size_t max_rep_;
    const int nodes_threshold_;
    const int lookahead_caching_;
    const double discount_;
    const bool debug_logs_;

    mutable size_t num_expansions_;
    mutable double total_time_;
    mutable double expand_time_;
    mutable size_t root_height_;
    mutable bool random_decision_;

    mutable size_t simulator_calls_;
    mutable double sim_time_;
    mutable double sim_reset_time_;

    mutable size_t get_atoms_calls_;
    mutable double get_atoms_time_;
    mutable double novel_atom_time_;
    mutable double update_novelty_time_;

    struct Node;
    Node* execution_node_;
    std::deque<Action> execution_branch_;

    virtual void do_solve(std::chrono::system_clock::time_point start_time,
                          Node* root,
                          std::unordered_map<int, FeatureVector > &novelty_table_map) =0;

    Action random_zero_value_action(const Node *root, double discount) const {
        assert(root != 0);
        assert((root->num_children_ > 0) && (root->first_child_ != nullptr));
        std::vector<Action> zero_value_actions;
        for( Node *child = root->first_child_; child != nullptr; child = child->sibling_ ) {
            if( child->qvalue(discount) == 0 )
                zero_value_actions.push_back(child->action_);
        }
        assert(!zero_value_actions.empty());
        return zero_value_actions[lrand48() % zero_value_actions.size()];
    }

    void call_simulator(const State& state, const Action& action, State& next_state, double& reward, bool& termination) {
        ++simulator_calls_;
        auto start_time = std::chrono::high_resolution_clock::now();
        domain_.compute_next_state(state, action);
        next_state = domain_.get_next_state(state, action);
        reward = domain_.get_transition_value(state, action, next_state);
        termination = domain_.is_terminal(next_state);
        assert(reward != -std::numeric_limits<double>::infinity());
        auto end_time = std::chrono::high_resolution_clock::now();
        sim_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / double(1e6);
    }

    void update_info(Node *node) {
        assert(node->is_info_valid_ != 2);
        assert((node->state_ == nullptr && node->parent_->state_ != nullptr) || // non-initial state node
               (node->state_ != nullptr && node->parent_->state_ == nullptr)); // initial state node
        assert(node->parent_ != nullptr);
        assert((node->parent_->is_info_valid_ == 1) || (node->parent_->state_ != nullptr) || (node->state_ != nullptr));
        if (!(node->parent_->state_)) { // initial state node
            assert(node->state_ != nullptr);
            node->terminal_ = domain_.is_terminal(*(node->state_));
            get_atoms(node);
            node->path_reward_ = 0;
            node->reward_ = 0;
        } else {
            double reward;
            bool termination;
            State state;
            call_simulator(*(node->parent_->state_), node->action_, state, reward, termination);
            assert(reward != std::numeric_limits<double>::infinity());
            assert(reward != -std::numeric_limits<double>::infinity());
            node->state_ = std::make_unique<State>(state);
            if( node->is_info_valid_ == 0 ) {
                node->reward_ = reward;
                node->terminal_ = termination;
                get_atoms(node);
                node->path_reward_ = node->parent_ == nullptr ? 0 : node->parent_->path_reward_;
                node->path_reward_ += node->reward_;
            }
        }
        node->is_info_valid_ = 2;
    }

    void get_atoms(const Node *node) const {
        assert(node->feature_atoms_.empty());
        ++get_atoms_calls_;
        auto start_time = std::chrono::high_resolution_clock::now();
        node->feature_atoms_ = this->state_to_feature_atoms_(*(node->state_));
        if( (node->parent_ != nullptr) && (typename FeatureVector::Equal()(node->parent_->feature_atoms_, node->feature_atoms_)) ) {
            node->frame_rep_ = node->parent_->frame_rep_ + this->frameskip_;
            assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        get_atoms_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / double(1e6);
    }

    // novelty tables: a (simple) novelty table maps feature indices to best depth at which
    // features have been seen. Best depth is initialized to max.int. Novelty table associated
    // to node is a unique simple table if subtables is disabled. Otherwise, there is one table
    // for each different logscore. The table for a node is the table for its logscore.
    int logscore(double path_reward) const {
        if( path_reward <= 0 ) {
            return 0;
        } else {
            int logr = int(floorf(log2f(path_reward)));
            return path_reward < 1 ? logr : 1 + logr;
        }
    }

    int get_index_for_novelty_table(const Node *node, bool use_novelty_subtables) const {
        return !use_novelty_subtables ? 0 : logscore(node->path_reward_);
    }

    FeatureVector& get_novelty_table(const Node *node, std::unordered_map<int, FeatureVector > &novelty_table_map, bool use_novelty_subtables) const {
        int index = get_index_for_novelty_table(node, use_novelty_subtables);
        typename std::unordered_map<int, FeatureVector >::iterator it = novelty_table_map.find(index);
        if( it == novelty_table_map.end() ) {
            novelty_table_map.insert(std::make_pair(index, FeatureVector()));
            FeatureVector &novelty_table = novelty_table_map.at(index);
            novelty_table = FeatureVector(num_tracked_atoms_, std::numeric_limits<int>::max());
            return novelty_table;
        } else {
            return it->second;
        }
    }

    size_t update_novelty_table(size_t depth, const FeatureVector &feature_atoms, FeatureVector &novelty_table) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t first_index = 0;
        size_t number_updated_entries = 0;
        for( size_t k = first_index; k < feature_atoms.size(); ++k ) {
            assert((feature_atoms[k] >= 0) && (feature_atoms[k] < int(novelty_table.size())));
            if( int(depth) < novelty_table[feature_atoms[k]] ) {
                novelty_table[feature_atoms[k]] = depth;
                ++number_updated_entries;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        update_novelty_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / double(1e6);
        return number_updated_entries;
    }

    int get_novel_atom(size_t depth, const FeatureVector &feature_atoms, const FeatureVector &novelty_table) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        for( size_t k = 0; k < feature_atoms.size(); ++k ) {
            assert(feature_atoms[k] < int(novelty_table.size()));
            if( novelty_table[feature_atoms[k]] > int(depth) ) {
                auto now_time = std::chrono::high_resolution_clock::now();
                novel_atom_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
                return feature_atoms[k];
            }
        }
        for( size_t k = 0; k < feature_atoms.size(); ++k ) {
            if( novelty_table[feature_atoms[k]] == int(depth) ) {
                auto now_time = std::chrono::high_resolution_clock::now();
                novel_atom_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
                return feature_atoms[k];
            }
        }
        auto now_time = std::chrono::high_resolution_clock::now();
        novel_atom_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
        assert(novelty_table[feature_atoms[0]] < int(depth));
        return feature_atoms[0];
    }

    size_t num_entries(const FeatureVector &novelty_table) const {
        assert(novelty_table.size() == num_tracked_atoms_);
        size_t n = 0;
        for( size_t k = 0; k < novelty_table.size(); ++k )
            n += novelty_table[k] < std::numeric_limits<int>::max();
        return n;
    }

    // generate states along given branch
    void generate_states_along_branch(Node *node, const std::deque<Action> &branch) {
        for( size_t pos = 0; pos < branch.size(); ++pos ) {
            if( node->state_ == nullptr ) {
                assert(node->is_info_valid_ == 1);
                update_info(node);
            }

            Node *selected = nullptr;
            for( Node *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
                if( typename Action::Equal()(child->action_, branch[pos]) ) {
                    selected = child;
                    break;
                }
            }
            assert(selected != nullptr);
            node = selected;
        }
    }

    virtual void reset_stats() {
        simulator_calls_ = 0;
        sim_time_ = 0;
        sim_reset_time_ = 0;
        update_novelty_time_ = 0;
        get_atoms_calls_ = 0;
        get_atoms_time_ = 0;
        novel_atom_time_ = 0;
        num_expansions_ = 0;
        total_time_ = 0;
        expand_time_ = 0;
        root_height_ = 0;
        random_decision_ = false;
    }

    virtual void print_stats(const Node &root, const std::unordered_map<int, FeatureVector > &novelty_table_map) const =0;

    class Node {
    public:
        bool visited_;                           // label
        bool solved_;                            // label
        Action action_;    // action mapping parent to this node
        int depth_;                              // node's depth
        int height_;                             // node's height (calculated)
        double reward_;                           // reward for this node
        double path_reward_;                      // reward of full path leading to this node
        int is_info_valid_;                      // is info valid? (0=no, 1=partial, 2=full)
        bool terminal_;                          // is node a terminal node?
        double value_;                            // backed up value

        mutable std::unique_ptr<State> state_;         // observation for this node
        mutable FeatureVector feature_atoms_; // features made true by this node
        mutable int num_novel_features_;         // number of features this node makes novel
        mutable int frame_rep_;                  // frame counter for number identical feature atoms through ancestors

        // structure
        int num_children_;                       // number of children
        Node *first_child_;                      // first child
        Node *sibling_;                          // right sibling of this node
        Node *parent_;                           // pointer to parent node

        Node(Node *parent, const Action& action, size_t depth)
        : visited_(false),
            solved_(false),
            action_(action),
            depth_(depth),
            height_(0),
            reward_(0),
            path_reward_(0),
            is_info_valid_(0),
            terminal_(false),
            value_(0),
            num_novel_features_(0),
            frame_rep_(0),
            num_children_(0),
            first_child_(nullptr),
            sibling_(nullptr),
            parent_(parent) {
        }
        ~Node() { }

        void remove_children() {
            while( first_child_ != nullptr ) {
                Node *child = first_child_;
                first_child_ = first_child_->sibling_;
                remove_tree(child);
            }
        }

        void expand(const Action& action) {
            Node *new_child = new Node(this, action, 1 + depth_);
            new_child->sibling_ = first_child_;
            first_child_ = new_child;
            ++num_children_;
        }
        void expand(Domain& domain, bool random_shuffle = true) {
            assert((num_children_ == 0) && (first_child_ == nullptr));
            auto applicable_actions = domain.get_applicable_actions(*state_)->get_elements();
            for (auto i = applicable_actions.begin(); i != applicable_actions.end() ; ++i) {
                expand(*i);
            }
            //if( random_shuffle ) std::random_shuffle(children_.begin(), children_.end()); // CHECK: missing
        }

        void clear_cached_observations() {
            if( is_info_valid_ == 2 ) {
                state_.reset();
                is_info_valid_ = 1;
            }
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                child->clear_cached_observations();
        }

        Node* advance(const Action& action) {
            assert((num_children_ > 0) && (first_child_ != nullptr));
            assert((parent_ == nullptr) || (parent_->parent_ == nullptr));
            if( parent_ != nullptr ) {
                delete parent_;
                parent_ = nullptr;
            }

            Node *selected = nullptr;
            Node *sibling = nullptr;
            for( Node *child = first_child_; child != nullptr; child = sibling ) {
                sibling = child->sibling_;
                if( typename Action::Equal()(child->action_, action) )
                    selected = child;
                else
                    IWSolver<Tdomain, Texecution_policy>::remove_tree(child);
            }
            assert(selected != nullptr);

            selected->sibling_ = nullptr;
            first_child_ = selected;
            return selected;
        }

        void normalize_depth(int depth = 0) {
            depth_ = depth;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                child->normalize_depth(1 + depth);
        }

        void reset_frame_rep_counters(int frameskip, int parent_frame_rep) {
            if( frame_rep_ > 0 ) {
                frame_rep_ = parent_frame_rep + frameskip;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                    child->reset_frame_rep_counters(frameskip, frame_rep_);
            }
        }
        void reset_frame_rep_counters(int frameskip) {
            reset_frame_rep_counters(frameskip, -frameskip);
        }

        void recompute_path_rewards(const Node *ref = nullptr) {
            if( this == ref ) {
                path_reward_ = 0;
            } else {
                assert(parent_ != nullptr);
                path_reward_ = parent_->path_reward_ + reward_;
            }
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                child->recompute_path_rewards();
        }

        void solve_and_backpropagate_label() {
            assert(!solved_);
            if( !solved_ ) {
                solved_ = true;
                if( parent_ != nullptr ) {
                    assert(!parent_->solved_);
                    bool unsolved_siblings = false;
                    for( Node *child = parent_->first_child_; child != nullptr; child = child->sibling_ ) {
                        if( !child->solved_ ) {
                            unsolved_siblings = true;
                            break;
                        }
                    }
                    if( !unsolved_siblings )
                        parent_->solve_and_backpropagate_label();
                }
            }
        }

        double qvalue(double discount) const {
            return reward_ + discount * value_;
        }

        double backup_values(double discount) {
            assert((num_children_ == 0) || (is_info_valid_ != 0));
            value_ = 0;
            if( num_children_ > 0 ) {
                assert(first_child_ != nullptr);
                double max_child_value = -std::numeric_limits<double>::infinity();
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    child->backup_values(discount);
                    double child_value = child->qvalue(discount);
                    max_child_value = std::max(max_child_value, child_value);
                }
                value_ = max_child_value;
            }
            return value_;
        }

        void best_branch(std::deque<Action> &branch, double discount) const {
            if( num_children_ > 0 ) {
                assert(first_child_ != nullptr);
                size_t num_best_children = 0;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                    num_best_children += child->qvalue(discount) == value_;
                assert(num_best_children > 0);
                size_t index_best_child = lrand48() % num_best_children;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    if( child->qvalue(discount) == value_ ) {
                        if( index_best_child == 0 ) {
                            branch.push_back(child->action_);
                            child->best_branch(branch, discount);
                            break;
                        }
                        --index_best_child;
                    }
                }
            }
        }

        void longest_zero_value_branch(double discount, std::deque<Action> &branch) const {
            assert(value_ == 0);
            if( num_children_ > 0 ) {
                assert(first_child_ != nullptr);
                size_t max_height = 0;
                size_t num_best_children = 0;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    if( (child->qvalue(discount) == 0) && (child->height_ >= int(max_height)) ) {
                        if( child->height_ > int(max_height) ) {
                            max_height = child->height_;
                            num_best_children = 0;
                        }
                        ++num_best_children;
                    }
                }
                assert(num_best_children > 0);
                size_t index_best_child = lrand48() % num_best_children;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    if( (child->qvalue(discount) == 0) && (child->height_ == int(max_height)) ) {
                        if( index_best_child == 0 ) {
                            branch.push_back(child->action_);
                            child->longest_zero_value_branch(discount, branch);
                            break;
                        }
                        --index_best_child;
                    }
                }
            }
        }

        size_t num_tip_nodes() const {
            if( num_children_ == 0 ) {
                return 1;
            } else {
                assert(first_child_ != nullptr);
                size_t n = 0;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                    n += child->num_tip_nodes();
                return n;
            }
        }

        size_t num_nodes() const {
            size_t n = 1;
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                n += child->num_nodes();
            return n;
        }

        int calculate_height() {
            height_ = 0;
            if( num_children_ > 0 ) {
                assert(first_child_ != nullptr);
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    int child_height = child->calculate_height();
                    height_ = std::max(height_, child_height);
                }
                height_ += 1;
            }
            return height_;
        }

        void print_branch(std::ostream &os, const std::deque<Action> &branch, size_t index = 0) const {
            print(os);
            if( index < branch.size() ) {
                Action action = branch[index];
                bool child_found = false;
                for( Node *child = first_child_; child != nullptr; child = child->sibling_ ) {
                    if( child->action_ == action ) {
                        child->print_branch(os, branch, ++index);
                        child_found = true;
                        break;
                    }
                }
                assert(child_found);
            }
        }

        void print(std::ostream &os) const {
            os << "node:"
            << " valid=" << is_info_valid_
            << ", solved=" << solved_
            << ", value=" << value_
            << ", reward=" << reward_
            << ", path-reward=" << path_reward_
            << ", action=" << action_
            << ", depth=" << depth_
            << ", children=[";
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                os << child->value_ << " ";
            os << "] (this=" << this << ", parent=" << parent_ << ")"
            << std::endl;
        }

        void print_tree(std::ostream &os) const {
            print(os);
            for( Node *child = first_child_; child != nullptr; child = child->sibling_ )
                child->print_tree(os);
        }
    };

    inline static void remove_tree(Node *node) {
        Node *sibling = nullptr;
        for( Node *child = node->first_child_; child != nullptr; child = sibling ) {
            sibling = child->sibling_;
            remove_tree(child);
        }
        delete node;
    }
};


template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class BfsIW: public IWSolver<Tdomain, Texecution_policy> {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef typename Domain::FeatureVector FeatureVector;
    typedef Texecution_policy ExecutionPolicy;

    BfsIW(Domain& domain,
          const std::function<FeatureVector (const State&)>& state_to_feature_atoms,
          size_t frameskip = 15,
          int simulator_budget = 150000,
          double time_budget = std::numeric_limits<double>::infinity(),
          bool novelty_subtables = false,
          bool random_actions = false,
          size_t max_rep = 30,
          int nodes_threshold = 50000,
          bool break_ties_using_rewards = false,
          double discount = 1.0,
          bool debug_logs = false)
        : IWSolver<Tdomain, Texecution_policy>(domain, state_to_feature_atoms, frameskip,
                                               simulator_budget, time_budget, novelty_subtables,
                                               random_actions, max_rep, nodes_threshold,
                                               discount, debug_logs),
          break_ties_using_rewards_(break_ties_using_rewards) {
        
    }

private :
    typedef typename IWSolver<Tdomain, Texecution_policy>::Node Node;
    struct NodeComparator;

    const bool break_ties_using_rewards_;

    virtual void do_solve(std::chrono::system_clock::time_point start_time,
                          Node* root,
                          std::unordered_map<int, FeatureVector > &novelty_table_map) {
        // priority queue
        NodeComparator cmp(break_ties_using_rewards_);
        std::priority_queue<Node*, std::vector<Node*>, NodeComparator> q(cmp);

        // add tip nodes to queue
        add_tip_nodes_to_queue(root, q);
        spdlog::info("queue: sz=" + std::to_string(q.size()));

        // explore in breadth-first manner
        auto now_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
        
        while( !q.empty() && (int(this->simulator_calls_) < this->simulator_budget_) && (elapsed_time < this->time_budget_) ) {
            bfs(root, q, novelty_table_map);
            auto now_time = std::chrono::high_resolution_clock::now();
            elapsed_time = double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
        }
    }

    void bfs(Node *root,
             std::priority_queue<Node*, std::vector<Node*>, NodeComparator>& q,
             std::unordered_map<int, FeatureVector > &novelty_table_map) {
        Node *node = q.top();
        q.pop();

        // print debug info
        spdlog::debug(std::to_string(node->depth_) + "@" + std::to_string(node->path_reward_));

        // update node info
        assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
        assert(node->visited_ || (node->is_info_valid_ != 2));
        if( node->is_info_valid_ != 2 ) {
            this->update_info(node);
            assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
            node->visited_ = true;
        }

        // check termination at this node
        if( node->terminal_ ) {
            spdlog::debug("t," );
            return;
        }

        // verify max repetitions of feature atoms (screen mode)
        if( node->frame_rep_ > int(this->max_rep_) ) {
            spdlog::debug("r" + std::to_string(node->frame_rep_) + ",");
            return;
        }

        // calculate novelty and prune
        if( node->frame_rep_ == 0 ) {
            // calculate novelty
            FeatureVector &novelty_table = this->get_novelty_table(node, novelty_table_map, this->novelty_subtables_);
            int atom = this->get_novel_atom(node->depth_, node->feature_atoms_, novelty_table);
            assert((atom >= 0) && (atom < int(novelty_table.size())));

            // prune node using novelty
            if( novelty_table[atom] <= node->depth_ ) {
                spdlog::debug("p,");
                return;
            }

            // update novelty table
            this->update_novelty_table(node->depth_, node->feature_atoms_, novelty_table);
        }
        spdlog::debug("+");

        // expand node
        if( node->frame_rep_ == 0 ) {
            ++(this->num_expansions_);
            auto start_time = std::chrono::high_resolution_clock::now();
            node->expand(this->domain_, false);
            auto now_time = std::chrono::high_resolution_clock::now();
            this->expand_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
        } else {
            assert(node->parent_ != nullptr);
            node->expand(node->action_);
        }
        assert((node->num_children_ > 0) && (node->first_child_ != nullptr));
        spdlog::debug(std::to_string(node->num_children_) + ",");

        // add children to queue
        for( Node *child = node->first_child_; child != nullptr; child = child->sibling_ )
            q.push(child);
    }

    void add_tip_nodes_to_queue(Node *node, std::priority_queue<Node*, std::vector<Node*>, NodeComparator> &pq) const {
        std::deque<Node*> q;
        q.push_back(node);
        while( !q.empty() ) {
            Node *n = q.front();
            q.pop_front();
            if( n->num_children_ == 0 ) {
                assert(n->first_child_ == nullptr);
                pq.push(n);
            } else {
                assert(n->first_child_ != nullptr);
                for( Node *child = n->first_child_; child != nullptr; child = child->sibling_ )
                    q.push_back(child);
            }
        }
    }

    virtual void print_stats(const Node &root, const std::unordered_map<int, FeatureVector > &novelty_table_map) const {
        std::string msg("decision-stats: #entries=[");

        for( typename std::unordered_map<int, FeatureVector >::const_iterator it = novelty_table_map.begin(); it != novelty_table_map.end(); ++it )
            msg += std::to_string(it->first) + ":" + std::to_string(this->num_entries(it->second)) + "/" + std::to_string(it->second.size()) + ",";

        msg +=  std::string("]")
            + " #nodes=" + std::to_string(root.num_nodes())
            + " #tips=" + std::to_string(root.num_tip_nodes())
            + " height=[" + std::to_string(root.height_) + ":";

        for( Node *child = root.first_child_; child != nullptr; child = child->sibling_ )
            msg += std::to_string(child->height_) + ",";

        msg += std::string("]")
            + " #expansions=" + std::to_string(this->num_expansions_)
            + " #sim=" + std::to_string(this->simulator_calls_)
            + " total-time=" + std::to_string(this->total_time_)
            + " simulator-time=" + std::to_string(this->sim_time_)
            + " reset-time=" + std::to_string(this->sim_reset_time_)
            + " expand-time=" + std::to_string(this->expand_time_)
            + " update-novelty-time=" + std::to_string(this->update_novelty_time_)
            + " get-atoms-calls=" + std::to_string(this->get_atoms_calls_)
            + " get-atoms-time=" + std::to_string(this->get_atoms_time_)
            + " novel-atom-time=" + std::to_string(this->novel_atom_time_);
        
        spdlog::info(msg);
    }

    // breadth-first search with ties broken in favor of bigger path reward
    struct NodeComparator {
        bool break_ties_using_rewards_;
        NodeComparator(bool break_ties_using_rewards) : break_ties_using_rewards_(break_ties_using_rewards) {
        }
        bool operator()(const Node *lhs, const Node *rhs) const {
            return
              (lhs->depth_ > rhs->depth_) ||
              (break_ties_using_rewards_ && (lhs->depth_ == rhs->depth_) && (lhs->path_reward_ < rhs->path_reward_));
        }
    };
};


template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class RolloutIW : public IWSolver<Tdomain, Texecution_policy> {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef typename Domain::FeatureVector FeatureVector;
    typedef Texecution_policy ExecutionPolicy;

    RolloutIW(Domain& domain,
              const std::function<FeatureVector (const State&)>& state_to_feature_atoms,
              size_t frameskip = 15,
              int simulator_budget = 150000,
              double time_budget = std::numeric_limits<double>::infinity(),
              bool novelty_subtables = false,
              bool random_actions = false,
              size_t max_rep = 30,
              int nodes_threshold = 50000,
              size_t max_depth = 1500,
              double discount = 1.0,
              bool debug_logs = false)
        : IWSolver<Tdomain, Texecution_policy>(domain, state_to_feature_atoms, frameskip,
                                               simulator_budget, time_budget, novelty_subtables,
                                               random_actions, max_rep, nodes_threshold,
                                               discount, debug_logs),
          max_depth_(max_depth) {
        
    }

private :
    typedef typename IWSolver<Tdomain, Texecution_policy>::Node Node;

    const size_t max_depth_;

    mutable size_t num_rollouts_;
    mutable size_t num_cases_[4];

    virtual void do_solve(std::chrono::system_clock::time_point start_time,
                          Node* root,
                          std::unordered_map<int, FeatureVector > &novelty_table_map) {
        // construct/extend lookahead tree
        auto now_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);

        // clear solved labels
        clear_solved_labels(root);
        root->parent_->solved_ = false;
        while( !root->solved_ && (int(this->simulator_calls_) < this->simulator_budget_) && (elapsed_time < this->time_budget_) ) {
            spdlog::debug("rollout-iw: new rollout");
            rollout(root, novelty_table_map);
            auto now_time = std::chrono::high_resolution_clock::now();
            elapsed_time = double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
        }
    }

    void rollout(Node *root, std::unordered_map<int, FeatureVector > &novelty_table_map) {
        ++num_rollouts_;

        // update root info
        if( root->is_info_valid_ != 2 )
            this->update_info(root);

        // perform rollout
        Node *node = root;
        while( !node->solved_ ) {
            assert(node->is_info_valid_ == 2);

            // if first time at this node, expand node
            expand_if_necessary(node);

            // pick random unsolved child
            node = pick_unsolved_child(node);
            assert(!node->solved_);

            // update info
            if( node->is_info_valid_ != 2 )
                this->update_info(node);

            // report non-zero rewards
            if( node->reward_ > 0 ) {
                spdlog::debug("rollout-iw: positive reward");
            } else if( node->reward_ < 0 ) {
                spdlog::debug("rollout-iw: negative reward");
            }

            // if terminal, label as solved and terminate rollout
            if( node->terminal_ ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "T[reward=" << node->reward_ << "]" << std::flush;
                break;
            }

            // verify repetitions of feature atoms (screen mode)
            if( node->frame_rep_ > int(this->max_rep_) ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "R" << std::flush;
                break;
            } else if( node->frame_rep_ > 0 ) {
                node->visited_ = true;
                //logos_ << "r" << std::flush;
                continue;
            }

            // calculate novelty
            FeatureVector &novelty_table = this->get_novelty_table(node, novelty_table_map, this->novelty_subtables_);
            int atom = this->get_novel_atom(node->depth_, node->feature_atoms_, novelty_table);
            assert((atom >= 0) && (atom < int(novelty_table.size())));

            // five cases
            if( node->depth_ > int(max_depth_) ) {
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "D" << std::flush;
                break;
            } else if( novelty_table[atom] > node->depth_ ) { // novel => not(visited)
                // when caching, the assertion
                //
                //   assert(!node->visited_);
                //
                // may be false as there may be nodes in given tree. We just replace
                // it by the if() expression below. Table updates are only peformed for
                // nodes added to existing tree.
                if( !node->visited_ ) {
                    ++num_cases_[0];
                    node->visited_ = true;
                    node->num_novel_features_ = this->update_novelty_table(node->depth_, node->feature_atoms_, novelty_table);
                    //logos_ << Utils::green() << "n" << Utils::normal() << std::flush;
                }
                continue;
            } else if( !node->visited_ && (novelty_table[atom] <= node->depth_) ) { // not(novel) and not(visited) => PRUNE
                ++num_cases_[1];
                node->visited_ = true;
                assert((node->num_children_ == 0) && (node->first_child_ == nullptr));
                node->solve_and_backpropagate_label();
                //logos_ << "x" << node->depth_ << std::flush;
                break;
            } else if( node->visited_ && (novelty_table[atom] < node->depth_) ) { // not(novel) and visited => PRUNE
                ++num_cases_[2];
                //node->remove_children();
                node->reward_ = -std::numeric_limits<double>::infinity();
                spdlog::debug("rollout-iw: negative reward");
                node->solve_and_backpropagate_label();
                //logos_ << "X" << node->depth_ << std::flush;
                break;
            } else { // optimal and visited => CONTINUE
                assert(node->visited_ && (novelty_table[atom] == node->depth_));
                ++num_cases_[3];
                //logos_ << "c" << std::flush;
                continue;
            }
        }
    }

    void expand_if_necessary(Node *node) const {
        if( node->num_children_ == 0 ) {
            assert(node->first_child_ == nullptr);
            if( node->frame_rep_ == 0 ) {
                ++(this->num_expansions_);
                auto start_time = std::chrono::high_resolution_clock::now();
                node->expand(this->domain_);
                auto now_time = std::chrono::high_resolution_clock::now();
                this->expand_time_ += double(std::chrono::duration_cast<std::chrono::microseconds>(now_time - start_time).count()) / double(1e6);
            } else {
                assert(node->parent_ != nullptr);
                node->expand(node->action_);
            }
            assert((node->num_children_ > 0) && (node->first_child_ != nullptr));
        }
    }

    Node* pick_unsolved_child(const Node *node) const {
        Node *selected = nullptr;

        // decide to pick among all unsolved children or among those
        // with biggest number of novel features
        bool filter_unsolved_children = false; //lrand48() % 2;

        // select unsolved child
        size_t num_candidates = 0;
        int novel_features_threshold = std::numeric_limits<int>::min();;
        for( Node *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
            if( !child->solved_ && (child->num_novel_features_ >= novel_features_threshold) ) {
                if( filter_unsolved_children && (child->num_novel_features_ > novel_features_threshold) ) {
                    novel_features_threshold = child->num_novel_features_;
                    num_candidates = 0;
                }
                ++num_candidates;
            }
        }
        assert(num_candidates > 0);
        size_t index = lrand48() % num_candidates;
        for( Node *child = node->first_child_; child != nullptr; child = child->sibling_ ) {
            if( !child->solved_ && (child->num_novel_features_ >= novel_features_threshold) ) {
                if( index == 0 ) {
                    selected = child;
                    break;
                }
                --index;
            }
        }
        assert(selected != nullptr);
        assert(!selected->solved_);
        return selected;
    }

    void clear_solved_labels(Node *node) const {
        node->solved_ = false;
        for( Node *child = node->first_child_; child != nullptr; child = child->sibling_ )
            clear_solved_labels(child);
    }

    virtual void reset_stats() {
        IWSolver<Tdomain, Texecution_policy>::reset_stats();
        num_rollouts_ = 0;
        num_cases_[0] = 0;
        num_cases_[1] = 0;
        num_cases_[2] = 0;
        num_cases_[3] = 0;
    }

    virtual void print_stats(const Node &root, const std::unordered_map<int, FeatureVector > &novelty_table_map) const {
        std::string msg = std::string("decision-stats: #rollouts=" + std::to_string(num_rollouts_) + " #entries=[");

        for( typename std::unordered_map<int, FeatureVector >::const_iterator it = novelty_table_map.begin(); it != novelty_table_map.end(); ++it )
            msg += std::to_string(it->first) + ":" + std::to_string(this->num_entries(it->second)) + "/" + std::to_string(it->second.size()) + ",";

        msg +=  std::string("]")
            + " #nodes=" + std::to_string(root.num_nodes())
            + " #tips=" + std::to_string(root.num_tip_nodes())
            + " height=[" + std::to_string(root.height_) + ":";

        for( Node *child = root.first_child_; child != nullptr; child = child->sibling_ )
            msg += std::to_string(child->height_) + ",";

        msg += std::string("]")
            + " #expansions=" + std::to_string(this->num_expansions_)
            + " #cases=[" + std::to_string(num_cases_[0]) + "," + std::to_string(num_cases_[1]) + "," + std::to_string(num_cases_[2]) + "," + std::to_string(num_cases_[3]) + "]"
            + " #sim=" + std::to_string(this->simulator_calls_)
            + " total-time=" + std::to_string(this->total_time_)
            + " simulator-time=" + std::to_string(this->sim_time_)
            + " reset-time=" + std::to_string(this->sim_reset_time_)
            + " expand-time=" + std::to_string(this->expand_time_)
            + " update-novelty-time=" + std::to_string(this->update_novelty_time_)
            + " get-atoms-calls=" + std::to_string(this->get_atoms_calls_)
            + " get-atoms-time=" + std::to_string(this->get_atoms_time_)
            + " novel-atom-time=" + std::to_string(this->novel_atom_time_);
        
        spdlog::info(msg);
    }
};

} // namespace airlaps

#endif // AIRLAPS_IW_HH
