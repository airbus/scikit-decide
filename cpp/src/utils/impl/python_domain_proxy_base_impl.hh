/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_IMPL_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_IMPL_HH

#include <pybind11/pybind11.h>

#include "utils/python_gil_control.hh"
#include "utils/python_globals.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === State implementation ===

#define SK_PY_STATE_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_STATE_CLASS PythonDomainProxyBase<Texecution>::State

#define SK_PY_STATE_TYPE typename PythonDomainProxyBase<Texecution>::State

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_CLASS::State() : PyObj<State>() {}

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_CLASS::State(std::unique_ptr<py::object> &&s)
    : PyObj<State>(std::move(s)) {}

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_CLASS::State(const py::object &s) : PyObj<State>(s) {}

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_CLASS::State(const State &other) : PyObj<State>(other) {}

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_TYPE &SK_PY_STATE_CLASS::operator=(const State &other) {
  static_cast<PyObj<State> &>(*this) = other;
  return *this;
}

SK_PY_STATE_TEMPLATE_DECL
SK_PY_STATE_CLASS::~State() {}

// === Observation implementation ===

#define SK_PY_OBSERVATION_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_OBSERVATION_CLASS PythonDomainProxyBase<Texecution>::Observation

#define SK_PY_OBSERVATION_TYPE                                                 \
  typename PythonDomainProxyBase<Texecution>::Observation

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_CLASS::Observation() : PyObj<Observation>() {}

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_CLASS::Observation(std::unique_ptr<py::object> &&o)
    : PyObj<Observation>(std::move(o)) {}

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_CLASS::Observation(const py::object &o)
    : PyObj<Observation>(o) {}

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_CLASS::Observation(const Observation &other)
    : PyObj<Observation>(other) {}

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_TYPE &
SK_PY_OBSERVATION_CLASS::operator=(const Observation &other) {
  static_cast<PyObj<Observation> &>(*this) = other;
  return *this;
}

SK_PY_OBSERVATION_TEMPLATE_DECL
SK_PY_OBSERVATION_CLASS::~Observation() {}

// === Event implementation ===

#define SK_PY_EVENT_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_EVENT_CLASS PythonDomainProxyBase<Texecution>::Event

#define SK_PY_EVENT_TYPE typename PythonDomainProxyBase<Texecution>::Event

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_CLASS::Event() : PyObj<Event>() {}

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_CLASS::Event(std::unique_ptr<py::object> &&e)
    : PyObj<Event>(std::move(e)) {}

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_CLASS::Event(const py::object &e) : PyObj<Event>(e) {}

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_CLASS::Event(const Event &other) : PyObj<Event>(other) {}

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_TYPE &SK_PY_EVENT_CLASS::operator=(const Event &other) {
  static_cast<PyObj<Event> &>(*this) = other;
  return *this;
}

SK_PY_EVENT_TEMPLATE_DECL
SK_PY_EVENT_CLASS::~Event() {}

// === Action implementation ===

#define SK_PY_ACTION_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_ACTION_CLASS PythonDomainProxyBase<Texecution>::Action

#define SK_PY_ACTION_TYPE typename PythonDomainProxyBase<Texecution>::Action

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_CLASS::Action() : PyObj<Action>() {}

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_CLASS::Action(std::unique_ptr<py::object> &&a)
    : PyObj<Action>(std::move(a)) {}

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_CLASS::Action(const py::object &a) : PyObj<Action>(a) {}

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_CLASS::Action(const Action &other) : PyObj<Action>(other) {}

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_TYPE &SK_PY_ACTION_CLASS::operator=(const Action &other) {
  static_cast<PyObj<Action> &>(*this) = other;
  return *this;
}

SK_PY_ACTION_TEMPLATE_DECL
SK_PY_ACTION_CLASS::~Action() {}

// === ApplicableActionSpaceElements implementation ===

#define SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL                   \
  template <typename Texecution>

#define SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS                           \
  PythonDomainProxyBase<Texecution>::ApplicableActionSpace::Elements

#define SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TYPE                            \
  typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace::Elements

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::Elements() : PyObj<Elements>() {}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::Elements(
    std::unique_ptr<py::object> &&applicable_action_space_elements)
    : PyObj<Elements>(std::move(applicable_action_space_elements)) {}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::Elements(
    const py::object &applicable_action_space_elements)
    : PyObj<Elements>(applicable_action_space_elements) {}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::Elements(const Elements &other)
    : PyObj<Elements>(other) {}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TYPE &
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::operator=(const Elements &other) {
  static_cast<PyObj<Elements> &>(*this) = other;
  return *this;
}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::~Elements() {}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
typename SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::PyIter
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::begin() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->begin());
}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
typename SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::PyIter
SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::end() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->end());
}

SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_TEMPLATE_DECL
bool SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS::empty() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (py::isinstance<py::list>(*(this->_pyobj))) {
      return py::cast<py::list>(*(this->_pyobj)).empty();
    } else if (py::isinstance<py::tuple>(*(this->_pyobj))) {
      return py::cast<py::tuple>(*(this->_pyobj)).empty();
    } else if (py::isinstance<py::dict>(*(this->_pyobj))) {
      return py::cast<py::dict>(*(this->_pyobj)).empty();
    } else if (py::isinstance<py::set>(*(this->_pyobj))) {
      return py::cast<py::set>(*(this->_pyobj)).empty();
    } else if (py::isinstance<py::sequence>(*(this->_pyobj))) {
      return py::cast<py::sequence>(*(this->_pyobj)).empty();
    } else {
      throw std::runtime_error("SKDECIDE exception: applicable action space "
                               "elements must be iterable.");
    }
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when checking emptiness of "
                              "applicable action space's elements: ") +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

// === ApplicableActionSpace implementation ===

#define SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL                            \
  template <typename Texecution>

#define SK_PY_APPLICABLE_ACTION_SPACE_CLASS                                    \
  PythonDomainProxyBase<Texecution>::ApplicableActionSpace

#define SK_PY_APPLICABLE_ACTION_SPACE_TYPE                                     \
  typename PythonDomainProxyBase<Texecution>::ApplicableActionSpace

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::ApplicableActionSpace()
    : PyObj<ApplicableActionSpace>() {
  construct();
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::ApplicableActionSpace(
    std::unique_ptr<py::object> &&applicable_action_space)
    : PyObj<ApplicableActionSpace>(std::move(applicable_action_space)) {
  construct();
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::ApplicableActionSpace(
    const py::object &applicable_action_space)
    : PyObj<ApplicableActionSpace>(applicable_action_space) {
  construct();
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
void SK_PY_APPLICABLE_ACTION_SPACE_CLASS::construct() {
  typename GilControl<Texecution>::Acquire acquire;
  if (this->_pyobj->is_none()) {
    this->_pyobj = std::make_unique<py::object>(
        skdecide::Globals::skdecide().attr("EmptySpace")());
  }
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::ApplicableActionSpace(
    const ApplicableActionSpace &other)
    : PyObj<ApplicableActionSpace>(other) {}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_TYPE &
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::operator=(
    const ApplicableActionSpace &other) {
  static_cast<PyObj<ApplicableActionSpace> &>(*this) = other;
  return *this;
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::~ApplicableActionSpace() {}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
typename SK_PY_APPLICABLE_ACTION_SPACE_ELEMENTS_CLASS
SK_PY_APPLICABLE_ACTION_SPACE_CLASS::get_elements() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!py::hasattr(*(this->_pyobj), "get_elements")) {
      throw std::invalid_argument(
          "SKDECIDE exception: python applicable action object must implement "
          "get_elements()");
    }
    return Elements(this->_pyobj->attr("get_elements")());
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting applicable "
                              "action space's elements: ") +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting applicable "
                              "action space's elements: ") +
                  std::string(e.what()));
    throw;
  }
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
bool SK_PY_APPLICABLE_ACTION_SPACE_CLASS::empty() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return py::isinstance(*(this->_pyobj),
                          skdecide::Globals::skdecide().attr("EmptySpace")) ||
           this->get_elements().empty();
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when checking emptyness of "
                              "applicable action space's elements: ") +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
SK_PY_ACTION_TYPE SK_PY_APPLICABLE_ACTION_SPACE_CLASS::sample() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!py::hasattr(*(this->_pyobj), "sample")) {
      throw std::invalid_argument("SKDECIDE exception: python applicable "
                                  "action object must implement sample()");
    } else {
      return Action(this->_pyobj->attr("sample")());
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when sampling action data: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(
        std::string("SKDECIDE exception when sampling action data: ") +
        std::string(e.what()));
    throw;
  }
}

SK_PY_APPLICABLE_ACTION_SPACE_TEMPLATE_DECL
bool SK_PY_APPLICABLE_ACTION_SPACE_CLASS::contains(const Action &action) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!py::hasattr(*(this->_pyobj), "contains")) {
      throw std::invalid_argument("SKDECIDE exception: python applicable "
                                  "action object must implement contains()");
    } else {
      return this->_pyobj->attr("contains")(action.pyobj())
          .template cast<bool>();
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string(
            "SKDECIDE exception when checking inclusion of action data: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

// === Value implementation ===

#define SK_PY_VALUE_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_VALUE_CLASS PythonDomainProxyBase<Texecution>::Value

#define SK_PY_VALUE_TYPE typename PythonDomainProxyBase<Texecution>::Value

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::Value() : PyObj<Value>() { construct(); }

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::Value(std::unique_ptr<py::object> &&v)
    : PyObj<Value>(std::move(v)) {
  construct();
}

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::Value(const py::object &v) : PyObj<Value>(v) { construct(); }

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::Value(const double &value, const bool &reward_or_cost)
    : PyObj<Value>() {
  construct(value, reward_or_cost);
}

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::Value(const Value &other) : PyObj<Value>(other) {}

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_TYPE &SK_PY_VALUE_CLASS::operator=(const Value &other) {
  static_cast<PyObj<Value> &>(*this) = other;
  return *this;
}

SK_PY_VALUE_TEMPLATE_DECL
SK_PY_VALUE_CLASS::~Value() {}

SK_PY_VALUE_TEMPLATE_DECL
void SK_PY_VALUE_CLASS::construct(const double &value,
                                  const bool &reward_or_cost) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (this->_pyobj->is_none()) {
      this->_pyobj = std::make_unique<py::object>(
          skdecide::Globals::skdecide().attr("Value")());
      if (reward_or_cost) {
        this->reward(value);
      } else {
        this->cost(value);
      }
    } else {
      if (!py::hasattr(*(this->_pyobj), "cost")) {
        throw std::invalid_argument("SKDECIDE exception: python value object "
                                    "must provide the 'cost' attribute");
      }
      if (!py::hasattr(*(this->_pyobj), "reward")) {
        throw std::invalid_argument("SKDECIDE exception: python value object "
                                    "must provide the 'reward' attribute");
      }
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when importing value data: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_VALUE_TEMPLATE_DECL
double SK_PY_VALUE_CLASS::cost() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return py::cast<double>(this->_pyobj->attr("cost"));
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when getting value's cost: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_VALUE_TEMPLATE_DECL
void SK_PY_VALUE_CLASS::cost(const double &c) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr("cost") = py::float_(c);
    this->_pyobj->attr("reward") = py::float_(-c);
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when setting value's cost: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_VALUE_TEMPLATE_DECL
double SK_PY_VALUE_CLASS::reward() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return this->_pyobj->attr("reward").template cast<double>();
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when getting value's reward: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_VALUE_TEMPLATE_DECL
void SK_PY_VALUE_CLASS::reward(const double &r) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr("reward") = py::float_(r);
    this->_pyobj->attr("cost") = py::float_(-r);
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when setting value's reward: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

// === Predicate implementation ===

#define SK_PY_PREDICATE_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_PREDICATE_CLASS PythonDomainProxyBase<Texecution>::Predicate

#define SK_PY_PREDICATE_TYPE                                                   \
  typename PythonDomainProxyBase<Texecution>::Predicate

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::Predicate() : PyObj<Predicate, py::bool_>() {
  construct();
}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::Predicate(std::unique_ptr<py::object> &&v)
    : PyObj<Predicate, py::bool_>(std::move(v)) {
  construct();
}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::Predicate(const py::object &v)
    : PyObj<Predicate, py::bool_>(v) {
  construct();
}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::Predicate(const bool &predicate)
    : PyObj<Predicate, py::bool_>(predicate) {}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::Predicate(const Predicate &other)
    : PyObj<Predicate, py::bool_>(other) {}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_TYPE &SK_PY_PREDICATE_CLASS::operator=(const Predicate &other) {
  static_cast<PyObj<Predicate, py::bool_> &>(*this) = other;
  return *this;
}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::~Predicate() {}

SK_PY_PREDICATE_TEMPLATE_DECL
void SK_PY_PREDICATE_CLASS::construct() {
  typename GilControl<Texecution>::Acquire acquire;
  if (this->_pyobj->is_none() || !py::isinstance<py::bool_>(*(this->_pyobj))) {
    this->_pyobj = std::make_unique<py::bool_>(false);
  }
}

SK_PY_PREDICATE_TEMPLATE_DECL
bool SK_PY_PREDICATE_CLASS::predicate() const {
  typename GilControl<Texecution>::Acquire acquire;
  return this->_pyobj->template cast<bool>();
}

SK_PY_PREDICATE_TEMPLATE_DECL
SK_PY_PREDICATE_CLASS::operator bool() const { return predicate(); }

SK_PY_PREDICATE_TEMPLATE_DECL
void SK_PY_PREDICATE_CLASS::predicate(const bool &p) {
  typename GilControl<Texecution>::Acquire acquire;
  *(this->_pyobj) = py::bool_(p);
}

SK_PY_PREDICATE_TEMPLATE_DECL
void SK_PY_PREDICATE_CLASS::operator=(const bool &p) { predicate(p); }

// === OutcomeInfo implementation ===

#define SK_PY_OUTCOME_INFO_TEMPLATE_DECL template <typename Texecution>

#define SK_PY_OUTCOME_INFO_CLASS PythonDomainProxyBase<Texecution>::OutcomeInfo

#define SK_PY_OUTCOME_INFO_TYPE                                                \
  typename PythonDomainProxyBase<Texecution>::OutcomeInfo

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_CLASS::OutcomeInfo() : PyObj<OutcomeInfo>() {}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_CLASS::OutcomeInfo(std::unique_ptr<py::object> &&s)
    : PyObj<OutcomeInfo>(std::move(s)) {}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_CLASS::OutcomeInfo(const py::object &s)
    : PyObj<OutcomeInfo>(s) {}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_CLASS::OutcomeInfo(const OutcomeInfo &other)
    : PyObj<OutcomeInfo>(other) {}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_TYPE &
SK_PY_OUTCOME_INFO_CLASS::operator=(const OutcomeInfo &other) {
  static_cast<PyObj<OutcomeInfo> &>(*this) = other;
  return *this;
}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
SK_PY_OUTCOME_INFO_CLASS::~OutcomeInfo() {}

SK_PY_OUTCOME_INFO_TEMPLATE_DECL
std::size_t SK_PY_OUTCOME_INFO_CLASS::get_depth() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (py::hasattr(*(this->_pyobj), "depth")) {
      return py::cast<std::size_t>(this->_pyobj->attr("depth")());
    } else {
      return 0;
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when getting outcome's depth info: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_BASE_IMPL_HH
