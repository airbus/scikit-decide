#ifndef AIRLAPS_PDDL_REQUIREMENTS_HH
#define AIRLAPS_PDDL_REQUIREMENTS_HH

namespace airlaps {

    namespace pddl {

        class Requirements {
        public :
            Requirements();
            Requirements(const Requirements& other);
            Requirements& operator=(const Requirements& other);
            
            Requirements& set_equality(bool equality = true);
            bool has_equality() const;

            Requirements& set_strips(bool strips = true);
            bool has_strips() const;

            Requirements& set_typing(bool typing = true);
            bool has_typing() const;

            Requirements& set_negative_preconditions(bool negative_preconditions = true);
            bool has_negative_preconditions() const;

            Requirements& set_disjunctive_preconditions(bool disjunctive_preconditions = true);
            bool has_disjunctive_preconditions() const;

            Requirements& set_existential_preconditions(bool existential_preconditions = true);
            bool has_existential_preconditions() const;

            Requirements& set_universal_preconditions(bool universal_preconditions = true);
            bool has_universal_preconditions() const;

            Requirements& set_conditional_effects(bool conditional_effects = true);
            bool has_conditional_effects() const;

            Requirements& set_fluents(bool fluents = true);
            bool has_fluents() const;

            Requirements& set_durative_actions(bool durative_actions = true);
            bool has_durative_actions() const;

            Requirements& set_time();
            bool has_time() const;

            Requirements& set_action_costs();
            bool has_action_costs() const;

            Requirements& set_object_fluents(bool object_fluents = true);
            bool has_object_fluents() const;

            Requirements& set_numeric_fluents(bool numeric_fluents = true);
            bool has_numeric_fluents() const;

            Requirements& set_modules(bool modules = true);
            bool has_modules() const;

            Requirements& set_adl();
            bool has_adl() const;

            Requirements& set_quantified_preconditions();
            bool has_quantified_preconditions() const;

            Requirements& set_duration_inequalities(bool duration_inequalities = true);
            bool has_duration_inequalities() const;

            Requirements& set_continuous_effects(bool continuous_effects = true);
            bool has_continuous_effects() const;

            Requirements& set_derived_predicates(bool derived_predicates = true);
            bool has_derived_predicates() const;

            Requirements& set_timed_initial_literals(bool timed_initial_literals = true);
            bool has_timed_initial_literals() const;

            Requirements& set_preferences(bool preferences = true);
            bool has_preferences() const;

            Requirements& set_constraints(bool constraints = true);
            bool has_constraints() const;

            std::string print() const;

        private :
            bool _equality;
            bool _strips;
            bool _typing;
            bool _negative_preconditions;
            bool _disjunctive_preconditions;
            bool _existential_preconditions;
            bool _universal_preconditions;
            bool _conditional_effects;
            bool _numeric_fluents;
            bool _object_fluents;
            bool _durative_actions;
            bool _time;
            bool _action_costs;
            bool _modules;
            bool _duration_inequalities;
            bool _continuous_effects;
            bool _derived_predicates;
            bool _timed_initial_literals;
            bool _preferences;
            bool _constraints;
        };

    } // namespace pddl

} // namespace airlaps

// Requirements printing operator
std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Requirements& r);

#endif // AIRLAPS_PDDL_REQUIREMENTS_HH
