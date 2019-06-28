#include "requirements.hh"

using namespace airlaps::pddl;

Requirements::Requirements()
: _equality(false),
  _strips(false),
  _typing(false),
  _negative_preconditions(false),
  _disjunctive_preconditions(false),
  _existential_preconditions(false),
  _universal_preconditions(false),
  _conditional_effects(false),
  _numeric_fluents(false),
  _object_fluents(false),
  _durative_actions(false),
  _time(false),
  _action_costs(false),
  _modules(false),
  _duration_inequalities(false),
  _continuous_effects(false),
  _derived_predicates(false),
  _timed_initial_literals(false),
  _preferences(false),
  _constraints(false) {

}


Requirements::Requirements(const Requirements& other)
: _equality(other._equality),
  _strips(other._strips),
  _typing(other._typing),
  _negative_preconditions(other._negative_preconditions),
  _disjunctive_preconditions(other._disjunctive_preconditions),
  _existential_preconditions(other._existential_preconditions),
  _universal_preconditions(other._universal_preconditions),
  _conditional_effects(other._conditional_effects),
  _numeric_fluents(other._numeric_fluents),
  _object_fluents(other._object_fluents),
  _durative_actions(other._durative_actions),
  _time(other._time),
  _action_costs(other._action_costs),
  _modules(other._modules),
  _duration_inequalities(other._duration_inequalities),
  _continuous_effects(other._continuous_effects),
  _derived_predicates(other._derived_predicates),
  _timed_initial_literals(other._timed_initial_literals),
  _preferences(other._preferences),
  _constraints(other._constraints) {

}


Requirements& Requirements::operator=(const Requirements& other) {
    _equality = other._equality;
    _strips = other._strips;
    _typing = other._typing;
    _negative_preconditions = other._negative_preconditions;
    _disjunctive_preconditions = other._disjunctive_preconditions;
    _existential_preconditions = other._existential_preconditions;
    _universal_preconditions = other._universal_preconditions;
    _conditional_effects = other._conditional_effects;
    _numeric_fluents = other._numeric_fluents;
    _object_fluents = other._object_fluents;
    _durative_actions = other._durative_actions;
    _time = other._time;
    _action_costs = other._action_costs;
    _modules = other._modules;
    _duration_inequalities = other._duration_inequalities;
    _continuous_effects = other._continuous_effects;
    _derived_predicates = other._derived_predicates;
    _timed_initial_literals = other._timed_initial_literals;
    _preferences = other._preferences;
    _constraints = other._constraints;
    return *this;
}


Requirements& Requirements::set_equality(bool equality) {
    _equality = equality;
    return *this;
}


bool Requirements::has_equality() const {
    return _equality;
}


Requirements& Requirements::set_strips(bool strips) {
    _strips = strips;
    return *this;
}


bool Requirements::has_strips() const {
    return _strips;
}

Requirements& Requirements::set_typing(bool typing) {
    _typing = typing;
    return *this;
}

bool Requirements::has_typing() const {
    return _typing;
}


Requirements& Requirements::set_negative_preconditions(bool negative_preconditions) {
    _negative_preconditions = negative_preconditions;
    return *this;
}


bool Requirements::has_negative_preconditions() const {
    return _negative_preconditions;
}


Requirements& Requirements::set_disjunctive_preconditions(bool disjunctive_preconditions) {
    _disjunctive_preconditions = disjunctive_preconditions;
    return *this;
}


bool Requirements::has_disjunctive_preconditions() const {
    return _disjunctive_preconditions;
}


Requirements& Requirements::set_existential_preconditions(bool existential_preconditions) {
    _existential_preconditions = existential_preconditions;
    return *this;
}


bool Requirements::has_existential_preconditions() const {
    return _existential_preconditions;
}


Requirements& Requirements::set_universal_preconditions(bool universal_preconditions) {
    _universal_preconditions = universal_preconditions;
    return *this;
}


bool Requirements::has_universal_preconditions() const {
    return _universal_preconditions;
}


Requirements& Requirements::set_conditional_effects(bool conditional_effects) {
    _conditional_effects = conditional_effects;
    return *this;
}


bool Requirements::has_conditional_effects() const {
    return _conditional_effects;
}


Requirements& Requirements::set_fluents(bool fluents) {
    _numeric_fluents = _object_fluents = fluents;
    return *this;
}


bool Requirements::has_fluents() const {
    return _numeric_fluents && _object_fluents;
}


Requirements& Requirements::set_durative_actions(bool durative_actions) {
    _durative_actions = durative_actions;
    return *this;
}


bool Requirements::has_durative_actions() const {
    return _durative_actions;
}


Requirements& Requirements::set_time() {
    _time = _numeric_fluents = _durative_actions = true;
    return *this;
}


bool Requirements::has_time() const {
    return _time && _numeric_fluents && _durative_actions;
}


Requirements& Requirements::set_action_costs() {
    _action_costs = _numeric_fluents = true;
    return *this;
}


bool Requirements::has_action_costs() const {
    return _action_costs && _numeric_fluents;
}


Requirements& Requirements::set_object_fluents(bool object_fluents) {
    _object_fluents = object_fluents;
    return *this;
}


bool Requirements::has_object_fluents() const {
    return _object_fluents;
}


Requirements& Requirements::set_numeric_fluents(bool numeric_fluents) {
    _numeric_fluents = numeric_fluents;
    return *this;
}


bool Requirements::has_numeric_fluents() const {
    return _numeric_fluents;
}


Requirements& Requirements::set_modules(bool modules) {
    _modules = modules;
    return *this;
}


bool Requirements::has_modules() const {
    return _modules;
}


Requirements& Requirements::set_adl() {
    _strips =
    _typing =
    _negative_preconditions =
    _disjunctive_preconditions =
    _equality =
    _existential_preconditions =
    _universal_preconditions =
    _conditional_effects = true;
    return *this;
}


bool Requirements::has_adl() const {
    return _strips &&
           _typing &&
           _negative_preconditions &&
           _disjunctive_preconditions &&
           _equality &&
           _existential_preconditions &&
           _universal_preconditions &&
           _conditional_effects;
}


Requirements& Requirements::set_quantified_preconditions() {
    _existential_preconditions = _universal_preconditions = true;
    return *this;
}


bool Requirements::has_quantified_preconditions() const {
    return _existential_preconditions && _universal_preconditions;
}


Requirements& Requirements::set_duration_inequalities(bool duration_inequalities) {
    _duration_inequalities = duration_inequalities;
    return *this;
}


bool Requirements::has_duration_inequalities() const {
    return _duration_inequalities;
}


Requirements& Requirements::set_continuous_effects(bool continuous_effects) {
    _continuous_effects = continuous_effects;
    return *this;
}


bool Requirements::has_continuous_effects() const {
    return _continuous_effects;
}


Requirements& Requirements::set_derived_predicates(bool derived_predicates) {
    _derived_predicates = derived_predicates;
    return *this;
}


bool Requirements::has_derived_predicates() const {
    return _derived_predicates;
}


Requirements& Requirements::set_timed_initial_literals(bool timed_initial_literals) {
    _timed_initial_literals = timed_initial_literals;
    return *this;
}


bool Requirements::has_timed_initial_literals() const {
    return _timed_initial_literals;
}


Requirements& Requirements::set_preferences(bool preferences) {
    _preferences = preferences;
    return *this;
}


bool Requirements::has_preferences() const {
    return _preferences;
}


Requirements& Requirements::set_constraints(bool constraints) {
    _constraints = constraints;
    return *this;
}


bool Requirements::has_constraints() const {
    return _constraints;
}
