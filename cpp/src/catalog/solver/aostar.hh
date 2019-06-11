#ifndef AIRLAPS_AOSTAR_HH
#define AIRLAPS_AOSTAR_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <queue>
#include <list>

#include "utils.hh"

namespace airlaps {

template <typename Tdomain>
class AOStarSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef std::shared_ptr<State> StatePtr;
    typedef typename Domain::Event Action;
    typedef std::shared_ptr<Action> ActionPtr;


    AOStarSolver(Domain& domain,
                 const std::function<bool (const State&)>& goal_checker,
                 const std::function<double (const State&)>& heuristic,
                 double discount = 1.0,
                 unsigned int max_tip_expansions = 1,
                 bool detect_cycles = false)
        : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
          _discount(discount), _max_tip_expansions(max_tip_expansions),
          _detect_cycles(detect_cycles) {}

    // reset the solver (clears the search graph, thus preventing from reusing
    // previous search results)
    void reset() {
        _graph.clear();
    }

    // solves from state s using heuristic function h
    void solve(const State& s) {
        auto si = _graph.emplace(s);
        if (si.first->solved || _goal_checker(s)) { // problem already solved from this state (was present in _graph and already solved)
            return;
        }
        StateNode& root_node = const_cast<StateNode&>(*(si.first)); // we won't change the real key (StateNode::state) so we are safe
        std::priority_queue<StateNode*, std::vector<StateNode*>, StateNodeCompare> q; // contains only non-goal unsolved tip nodes
        q.push(&root_node);

        while (!q.empty()) {
            unsigned int nb_expansions = std::min((unsigned int) q.size(), _max_tip_expansions);
            std::unordered_set<StateNode*> frontier;
            for (unsigned int cnt = 0 ; cnt < nb_expansions ; cnt++) {
                // Select best tip node of best partial graph
                StateNode* best_tip_node = q.top();
                q.pop();
                frontier.insert(best_tip_node);

                // Expand best tip node
                for (const auto& a : _domain.get_applicable_actions(best_tip_node->state)->get_elements()) {
                    best_tip_node->actions.push_back(std::make_unique<ActionNode>(a));
                    ActionNode& an = *(best_tip_node->actions.back());
                    an.parent = best_tip_node;
                    for (const auto& ns : _domain.get_next_state_distribution(best_tip_node->state, a)->get_values()) {
                        typename Domain::OutcomeExtractor oe(ns);
                        auto i = _graph.emplace(oe.state());
                        StateNode& next_node = const_cast<StateNode&>(*(i.first)); // we won't change the real key (StateNode::state) so we are safe
                        an.outcomes.push_back(std::make_tuple(oe.probability(), _domain.get_transition_value(best_tip_node->state, a, next_node.state), &next_node));
                        next_node.parents.push_back(&an);
                        if (i.second) { // new node
                            if (_goal_checker(next_node.state)) {
                                next_node.solved = true;
                                next_node.best_value = 0.0;
                            } else {
                                next_node.best_value = _heuristic(next_node.state);
                            }
                        }
                    }
                }
            }

            // Back-propagate value function from best tip node
            std::unique_ptr<std::unordered_set<StateNode*>> explored_states; // only for detecting cycles
            if (_detect_cycles) explored_states = std::make_unique<std::unordered_set<StateNode*>>(frontier);
            while (!frontier.empty()) {
                std::unordered_set<StateNode*> new_frontier;
                for (const auto& fs : frontier) {
                    // update Q-values and V-value
                    fs->best_value = std::numeric_limits<double>::infinity();
                    fs->best_action = nullptr;
                    for (const auto& a : fs->actions) {
                        a->value = 0.0;
                        for (const auto& ns : a->outcomes) {
                            a->value += std::get<0>(ns) * (std::get<1>(ns) + (_discount * std::get<2>(ns)->best_value));
                        }
                        if (a->value < fs->best_value) {
                            fs->best_value = a->value;
                            fs->best_action = a.get();
                        }
                        fs->best_value = std::min(fs->best_value, a->value);
                    }
                    // update solved field
                    fs->solved = true;
                    for (const auto& ns : fs->best_action->outcomes) {
                        fs->solved = fs->solved && std::get<2>(ns)->solved;
                    }
                    // update new frontier
                    for (const auto& ps : fs->parents) {
                        new_frontier.insert(ps->parent);
                    }
                }
                frontier = new_frontier;
                if (_detect_cycles) {
                    for (const auto& ps : new_frontier) {
                        if (explored_states->find(ps) != explored_states->end()) {
                            throw std::logic_error("AIRLAPS exception: cycle detected in the MDP graph! [with state " + ps->state.print() + "]");
                        }
                        explored_states->insert(ps);
                    }
                }
            }

            // Recompute best partial graph
            q = std::priority_queue<StateNode*, std::vector<StateNode*>, StateNodeCompare>();
            frontier.insert(&root_node);
            while (!frontier.empty()) {
                std::unordered_set<StateNode*> new_frontier;
                for (const auto& fs : frontier) {
                    if (!(fs->solved)) {
                        if (fs->best_action != nullptr) {
                            for (const auto& ns : fs->best_action->outcomes) {
                                new_frontier.insert(std::get<2>(ns));
                            }
                        } else { // tip node
                            q.push(fs);
                        }
                    }
                }
                frontier = new_frontier;
            }
        }
    }

    const Action& get_best_action(const State& s) const {
        auto si = _graph.find(s);
        if ((si == _graph.end()) || (si->best_action == nullptr)) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->best_action->action;
    }

    const double& get_best_value(const State& s) const {
        auto si = _graph.find(s);
        if (si == _graph.end()) {
            throw std::runtime_error("AIRLAPS exception: no best action found in state " + s.print());
        }
        return si->best_value;
    }

private :
    Domain& _domain;
    std::function<bool (const State&)> _goal_checker;
    std::function<double (const State&)> _heuristic;
    double _discount;
    unsigned int _max_tip_expansions;
    bool _detect_cycles;

    struct ActionNode;

    struct StateNode {
        State state;
        std::list<std::unique_ptr<ActionNode>> actions;
        ActionNode* best_action;
        double best_value;
        bool solved;
        std::list<ActionNode*> parents;

        StateNode(const State& s)
            : state(s), best_action(nullptr),
              best_value(std::numeric_limits<double>::infinity()),
              solved(false) {}
        
        struct Key {
            const State& operator()(const StateNode& sn) const { return sn.state; }
        };
    };

    struct ActionNode {
        Action action;
        std::list<std::tuple<double, double, StateNode*>> outcomes; // next state nodes owned by _graph
        double value;
        StateNode* parent;

        ActionNode(const Action& a)
            : action(a), value(std::numeric_limits<double>::infinity()), parent(nullptr) {}
    };

    struct StateNodeCompare {
        bool operator()(StateNode*& a, StateNode*& b) const {
            return (a->best_value) > (b->best_value); // smallest element appears at the top of the priority_queue => cost optimization
        }
    };

    typename SetTypeDeducer<StateNode, State>::Set _graph;
};

} // namespace airlaps

#endif // AIRLAPS_AOSTAR_HH
