/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
//
// Created by TEICHTEIL_FL on 15/05/2018.
//

#ifndef SKDECIDE_ASTAR_PLANNER_HH
#define SKDECIDE_ASTAR_PLANNER_HH


#include <solver/cpp/planner.hh>
#include <mutex>
#include <set>
#include <map>
#include <unordered_map>
#include <queue>
#include <iostream>


/**
 * Class representing a reentrant A* planner with ability to safely call the planner from different states in parallel
 * @tparam Tmodel Type of the underlying deterministic planning model (must derive from DeterministicPlanning<...>)
 * @tparam TplanContainer Type of a solution plan container
 */
template <typename Tmodel, template<typename...> typename TplanContainer = std::list>
class AstarPlanner : public DeterministicPlanner<Tmodel, TplanContainer<typename Tmodel::action_type> > {

public :

    /**
     * Type of a solution plan
     */
    typedef TplanContainer<typename Tmodel::action_type> plan_type;

    /**
     * Constructor
     * @param m Deterministic planning model
     * @param h Heuristic function
     */
    AstarPlanner(Tmodel& m, const std::function<typename Tmodel::cost_type(const typename Tmodel::state_type&, const typename Tmodel::state_set_type)>& h)
            : DeterministicPlanner<Tmodel, plan_type>(m), m_heuristic(h) {}

    /**
     * Reset the planner, which means that further calls to plan_from_path or plan_from_state won't reuse the previous computations
     */
    virtual void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_plans.clear();
        m_costs.clear();
    }

    /**
     * Run the planning algorithm from a given state
     * @param s Current state from which to launch the planning algorithm
     * @param f Functor called by the planning algorithm during optimization at regular time interval (inputs: current best plan, current best value; output: boolean which is true if the algorithm should stop now)
     */
    virtual void plan_from_state(const typename Tmodel::state_type& s, const std::function<bool(const plan_type&, const typename Tmodel::cost_type&)>& f = nullptr) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_plans[s].clear();
            m_costs[s] = typename Tmodel::cost_type();
        }

        priority_queue_type open_queue; // Priority queue used to sort states by increasing cost-to-go values (so-called OPEN container)
        typename Tmodel::state_set_type closed_set; // set of states for which the g-value is optimal (so-camlled CLOSED container)
        cost_map_type gscore; // Mapping from states to their distances from s
        state_map_type came_from; // Mapping from a given state to the state where it is currently optimal to come from
        
        open_queue.push(std::make_pair(s, m_heuristic(s, this->m_model.get_goal_states())));

        while (!open_queue.empty()) {
            auto current_state = open_queue.top().first;
            open_queue.pop();

            if (this->m_model.get_goal_states().find(current_state) != this->m_model.get_goal_states().end()) {
                std::lock_guard<std::mutex> lock(m_mutex);
                plan_type& plan = m_plans[s];
                typename Tmodel::cost_type& cost = m_costs[s];

                while (current_state != s) {
                    auto p = came_from[current_state];
                    cost += this->m_model.get_transition_cost(p.first, p.second, current_state);
                    current_state = p.first;
                    plan.push_front(p.second);
                }
                
                if (f != nullptr) {
                    f(plan, cost);
                }

                return;
            }

            closed_set.insert(current_state);
            auto applicable_actions = this->m_model.get_applicable_action_set(current_state);

            for (auto i = applicable_actions.begin() ; i != applicable_actions.end() ; ++i) {
                typename Tmodel::state_type neighbor = this->m_model.get_next_state(current_state, *i);

                if (closed_set.find(neighbor) != closed_set.end()) {
                    // Ignore the neighbor which is already evaluated
                    continue;
                }

                typename Tmodel::cost_type tentative_gscore = gscore[current_state] + (this->m_model.get_transition_cost(current_state, *i, neighbor));

                if ((gscore.find(neighbor) == gscore.end()) || (tentative_gscore < gscore[neighbor])) {
                    gscore[neighbor] = tentative_gscore;
                    came_from[neighbor] = std::make_pair(current_state, *i);
                    open_queue.push(std::make_pair(neighbor, tentative_gscore + m_heuristic(neighbor, this->m_model.get_goal_states())));
                }
            }
        }
    }

    /**
    * Get the value of the best found plan from a given state
    * @param s Current state
    * @return Value of the best found plan as a cost
    */
    virtual typename Tmodel::cost_type get_cost_from_state(const typename Tmodel::state_type& s) {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_costs[s];
    }

    /**
     * Get the best found plan from a given state
     * @param a Current state
     * @return Best found policy
     */
    virtual plan_type get_plan_from_state(const typename Tmodel::state_type& s) {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_plans[s];
    }

protected :

    /**
     * Automatically deduces the type of mapping from states to plans depending on state_set_type
     */
    typedef typename std::conditional<std::is_base_of<std::set<typename Tmodel::state_type>, typename Tmodel::state_set_type>::value,
                                     std::map<typename Tmodel::state_type, plan_type>,
                                     std::unordered_map<typename Tmodel::state_type, plan_type> >::type plan_map_type;

    /**
     * Automatically deduces the type of mapping from states to costs values depending on state_set_type
     */
    typedef typename std::conditional<std::is_base_of<std::set<typename Tmodel::state_type>, typename Tmodel::state_set_type>::value,
                                      std::map<typename Tmodel::state_type, typename Tmodel::cost_type>,
                                      std::unordered_map<typename Tmodel::state_type, typename Tmodel::cost_type> >::type cost_map_type;

    /**
     * Automatically deduces the type of mapping from states to states values depending on state_set_type
     */
    typedef typename std::conditional<std::is_base_of<std::set<typename Tmodel::state_type>, typename Tmodel::state_set_type>::value,
                                      std::map<typename Tmodel::state_type, std::pair<typename Tmodel::state_type, typename Tmodel::action_type> >,
                                      std::unordered_map<typename Tmodel::state_type, std::pair<typename Tmodel::state_type, typename Tmodel::action_type> > >::type state_map_type;

    /**
     * Heuristic function
     */
    std::function<typename Tmodel::cost_type(const typename Tmodel::state_type&, const typename Tmodel::state_set_type)> m_heuristic;

    /**
     * Mutex to protect concurrent parallel access to solution plans (m_plans) and costs (m_costs)
     */
    std::mutex m_mutex;

    /**
     * State comparison operator based on cost-to-go values
     */
    struct state_cost_cmp {
        typename Tmodel::cost_type operator()(const std::pair<typename Tmodel::state_type, typename Tmodel::cost_type>& left,
                                              const std::pair<typename Tmodel::state_type, typename Tmodel::cost_type>& right) const {
            return left.second > right.second;
        }
    };

    /**
     * Type of the priority queue used to sort states by increasing cost-to-go values
     */
    typedef std::priority_queue<std::pair<typename Tmodel::state_type, typename Tmodel::cost_type>,
                                std::vector<std::pair<typename Tmodel::state_type, typename Tmodel::cost_type> >,
                                state_cost_cmp> priority_queue_type;

    /**
     * Solution plans
     */
    plan_map_type m_plans;

    /**
     * Solution costs
     */
    cost_map_type m_costs;

};


#endif //SKDECIDE_ASTAR_PLANNER_HH
