/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_IMPL_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_IMPL_HH

#include <nngpp/nngpp.h>
#include <nngpp/protocol/pull0.h>

#include <pybind11/pybind11.h>

#include "utils/python_gil_control.hh"
#include "utils/python_globals.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === Agent implementation ===

#define SK_PY_AGENT_TEMPLATE_DECL                                              \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_AGENT_CLASS                                                      \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::Agent

#define SK_PY_AGENT_TYPE                                                       \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::Agent

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_CLASS::Agent() : PyObj<Agent>() {}

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_CLASS::Agent(std::unique_ptr<py::object> &&a)
    : PyObj<Agent>(std::move(a)) {}

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_CLASS::Agent(const py::object &a) : PyObj<Agent>(a) {}

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_CLASS::Agent(const Agent &other) : PyObj<Agent>(other) {}

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_TYPE &SK_PY_AGENT_CLASS::operator=(const Agent &other) {
  static_cast<PyObj<Agent> &>(*this) = other;
  return *this;
}

SK_PY_AGENT_TEMPLATE_DECL
SK_PY_AGENT_CLASS::~Agent() {}

// === AgentDataAccess implementation ===

#define SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL                                  \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>                       \
  template <typename DData, typename TTagent>

#define SK_PY_AGENT_DATA_ACCESS_CLASS                                          \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::                                                 \
      AgentDataAccess<DData, TTagent,                                          \
                      typename std::enable_if<                                 \
                          std::is_same<TTagent, MultiAgent>::value>::type>

#define SK_PY_AGENT_DATA_ACCESS_TYPE                                           \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::                      \
      template AgentDataAccess<DData, TTagent,                                 \
                               typename std::enable_if<std::is_same<           \
                                   TTagent, MultiAgent>::value>::type>

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_CLASS::AgentDataAccess()
    : PyObj<AgentData, py::dict>() {}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_CLASS::AgentDataAccess(std::unique_ptr<py::object> &&ad)
    : PyObj<AgentData, py::dict>(std::move(ad)) {}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_CLASS::AgentDataAccess(const py::object &ad)
    : PyObj<AgentData, py::dict>(ad) {}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_CLASS::AgentDataAccess(const AgentDataAccess &other)
    : PyObj<AgentData, py::dict>(other) {}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_TYPE &
SK_PY_AGENT_DATA_ACCESS_CLASS::operator=(const AgentDataAccess &other) {
  static_cast<PyObj<AgentData, py::dict> &>(*this) = other;
  return *this;
}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_CLASS::~AgentDataAccess() {}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
std::size_t SK_PY_AGENT_DATA_ACCESS_CLASS::size() const {
  typename GilControl<Texecution>::Acquire acquire;
  return this->_pyobj->size();
}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_TYPE::AgentDataAccessor
SK_PY_AGENT_DATA_ACCESS_CLASS::operator[](const Agent &a) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!(this->_pyobj->contains(a.pyobj()))) {
      (*(this->_pyobj))[a.pyobj()] = AgentData().pyobj();
    }
    return AgentDataAccessor((*(this->_pyobj))[a.pyobj()]);
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  AgentData::class_name + " of agent " + a.print() + ": " +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
const SK_PY_AGENT_DATA_ACCESS_TYPE::AgentData
SK_PY_AGENT_DATA_ACCESS_CLASS::operator[](const Agent &a) const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!(this->_pyobj->contains(a.pyobj()))) {
      throw std::runtime_error(std::string("SKDECIDE exception when getting ") +
                               AgentData::class_name + " of agent " +
                               a.print() + ": agent not in dictionary");
    }
    return AgentData((*(this->_pyobj))[a.pyobj()]);
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  AgentData::class_name + " of agent " + a.print() + ": " +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_TYPE::PyIter
SK_PY_AGENT_DATA_ACCESS_CLASS::begin() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->begin());
}

SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_TYPE::PyIter
SK_PY_AGENT_DATA_ACCESS_CLASS::end() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->end());
}

// === AgentDataAccess::Item implementation ===

#define SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL                                    \
  SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL

#define SK_PY_AGENT_DATA_ITEM_CLASS SK_PY_AGENT_DATA_ACCESS_CLASS::Item

#define SK_PY_AGENT_DATA_ITEM_TYPE SK_PY_AGENT_DATA_ACCESS_TYPE::Item

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_CLASS::Item() : PyObj<Item, py::tuple>() {}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_CLASS::Item(std::unique_ptr<py::object> &&a)
    : PyObj<Item, py::tuple>(std::move(a)) {}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_CLASS::Item(const py::object &a)
    : PyObj<Item, py::tuple>(a) {}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_CLASS::Item(const Item &other)
    : PyObj<Item, py::tuple>(other) {}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_TYPE &
SK_PY_AGENT_DATA_ITEM_CLASS::operator=(const Item &other) {
  static_cast<PyObj<Item, py::tuple> &>(*this) = other;
  return *this;
}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ITEM_CLASS::~Item() {}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_TYPE SK_PY_AGENT_DATA_ITEM_CLASS::agent() {
  typename GilControl<Texecution>::Acquire acquire;
  return Agent(py::cast<py::object>((*(this->_pyobj))[0]));
}

SK_PY_AGENT_DATA_ITEM_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESS_TYPE::AgentData SK_PY_AGENT_DATA_ITEM_CLASS::data() {
  typename GilControl<Texecution>::Acquire acquire;
  return AgentData(py::cast<py::object>((*(this->_pyobj))[1]));
}

// === AgentDataAccess::AgentDataAccessor implementation ===

#define SK_PY_AGENT_DATA_ACCESSOR_TEMPLATE_DECL                                \
  SK_PY_AGENT_DATA_ACCESS_TEMPLATE_DECL

#define SK_PY_AGENT_DATA_ACCESSOR_CLASS                                        \
  SK_PY_AGENT_DATA_ACCESS_CLASS::AgentDataAccessor

#define SK_PY_AGENT_DATA_ACCESSOR_TYPE                                         \
  SK_PY_AGENT_DATA_ACCESS_TYPE::AgentDataAccessor

SK_PY_AGENT_DATA_ACCESSOR_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESSOR_CLASS::AgentDataAccessor(
    const py::detail::item_accessor &a)
    : PyObj<AgentDataAccessor, py::detail::item_accessor>(a), AgentData(a) {}

SK_PY_AGENT_DATA_ACCESSOR_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESSOR_TYPE &
SK_PY_AGENT_DATA_ACCESSOR_CLASS::operator=(AgentDataAccessor &&other) && {
  std::forward<AgentDataAccessor &&>(*this) =
      static_cast<const AgentData &>(other);
  return *this;
}

SK_PY_AGENT_DATA_ACCESSOR_TEMPLATE_DECL
void SK_PY_AGENT_DATA_ACCESSOR_CLASS::operator=(const AgentData &other) && {
  typename GilControl<Texecution>::Acquire acquire;
  std::forward<py::detail::item_accessor &&>(
      *PyObj<AgentDataAccessor, py::detail::item_accessor>::_pyobj) =
      other.pyobj();
  *AgentData::_pyobj = other.pyobj();
}

SK_PY_AGENT_DATA_ACCESSOR_TEMPLATE_DECL
SK_PY_AGENT_DATA_ACCESSOR_CLASS::~AgentDataAccessor() {}

// === MemoryState implementation ===

#define SK_PY_MEMORY_STATE_TEMPLATE_DECL                                       \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_MEMORY_STATE_CLASS                                               \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::MemoryState

#define SK_PY_MEMORY_STATE_TYPE                                                \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::MemoryState

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_CLASS::MemoryState() : PyObj<MemoryState, py::list>() {}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_CLASS::MemoryState(std::unique_ptr<py::object> &&m)
    : PyObj<MemoryState, py::list>(std::move(m)) {}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_CLASS::MemoryState(const py::object &m)
    : PyObj<MemoryState, py::list>(m) {}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_CLASS::MemoryState(const MemoryState &other)
    : PyObj<MemoryState, py::list>(other) {}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_TYPE &
SK_PY_MEMORY_STATE_CLASS::operator=(const MemoryState &other) {
  static_cast<PyObj<MemoryState, py::list> &>(*this) = other;
  return *this;
}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_CLASS::~MemoryState() {}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
void SK_PY_MEMORY_STATE_CLASS::push_state(const State &s) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_pyobj->append(s.pyobj());
}

SK_PY_MEMORY_STATE_TEMPLATE_DECL
SK_PY_MEMORY_STATE_TYPE::State SK_PY_MEMORY_STATE_CLASS::last_state() {
  typename GilControl<Texecution>::Acquire acquire;
  if (this->_pyobj->empty()) {
    throw std::runtime_error("Cannot get last state of empty memory state " +
                             this->print());
  } else {
    return State((*(this->_pyobj))[this->_pyobj->size() - 1]);
  }
}

// === Outcome implementation ===

#define SK_PY_OUTCOME_TEMPLATE_DECL                                            \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>                       \
  template <typename Derived, typename Situation>

#define SK_PY_OUTCOME_CLASS                                                    \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::Outcome<Derived, Situation>

#define SK_PY_OUTCOME_TYPE                                                     \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability,                                 \
                             Tmemory>::template Outcome<Derived, Situation>

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::Outcome() : PyObj<Derived>() { construct(); }

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::Outcome(std::unique_ptr<py::object> &&outcome)
    : PyObj<Derived>(std::move(outcome)) {
  construct();
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::Outcome(const py::object &outcome)
    : PyObj<Derived>(outcome) {
  construct();
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::Outcome(const Situation &situation,
                             const Value &transition_value,
                             const Predicate &termination, const Info &info)
    : PyObj<Derived>() {
  construct(situation, transition_value, termination, info);
}

SK_PY_OUTCOME_TEMPLATE_DECL
void SK_PY_OUTCOME_CLASS::construct(const Situation &situation,
                                    const Value &transition_value,
                                    const Predicate &termination,
                                    const Info &info) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (this->_pyobj->is_none()) {
      this->_pyobj =
          std::make_unique<py::object>(skdecide::Globals::skdecide().attr(
              Derived::pyclass)(situation.pyobj()));
      this->transition_value(transition_value);
      this->termination(termination);
      this->info(info);
    } else {
      if (!py::hasattr(*(this->_pyobj), Derived::situation_name)) {
        throw std::invalid_argument(
            std::string("SKDECIDE exception: python ") + Derived::class_name +
            " object must provide '" + Derived::situation_name + "'");
      }
      if (!py::hasattr(*(this->_pyobj), "value")) {
        throw std::invalid_argument(std::string("SKDECIDE exception: python ") +
                                    Derived::class_name +
                                    " object must provide 'value'");
      }
      if (!py::hasattr(*(this->_pyobj), "termination")) {
        throw std::invalid_argument(std::string("SKDECIDE exception: python ") +
                                    Derived::class_name +
                                    " object must provide 'termination'");
      }
      if (!py::hasattr(*(this->_pyobj), "info")) {
        throw std::invalid_argument(std::string("SKDECIDE exception: python ") +
                                    Derived::class_name +
                                    " object must provide 'info'");
      }
    }
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when importing ") +
                  Derived::class_name + " data: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::Outcome(const Outcome &other) : PyObj<Derived>(other) {}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_TYPE &SK_PY_OUTCOME_CLASS::operator=(const Outcome &other) {
  static_cast<PyObj<Derived> &>(*this) = other;
  return *this;
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_CLASS::~Outcome() {}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_TYPE::Situation SK_PY_OUTCOME_CLASS::situation() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return Situation(this->_pyobj->attr(Derived::situation_name));
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name + "'s " + Derived::situation_name + ": " +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name + "'s " + Derived::situation_name + ": " +
                  std::string(e.what()));
    throw;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
void SK_PY_OUTCOME_CLASS::situation(const Situation &s) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr(Derived::situation_name) = s.pyobj();
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when setting ") +
                  Derived::class_name + "'s " + Derived::situation_name + ": " +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_TYPE::Value SK_PY_OUTCOME_CLASS::transition_value() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return Value(this->_pyobj->attr("value"));
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name +
                  "'s transition value: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name +
                  "'s transition value: " + std::string(e.what()));
    throw;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
void SK_PY_OUTCOME_CLASS::transition_value(const Value &tv) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr("value") = tv.pyobj();
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when setting outcome's value: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_TYPE::Predicate SK_PY_OUTCOME_CLASS::termination() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return Predicate(this->_pyobj->attr("termination"));
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name +
                  "'s termination: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name +
                  "'s termination: " + std::string(e.what()));
    throw;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
void SK_PY_OUTCOME_CLASS::termination(const Predicate &t) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr("termination") = t.pyobj();
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when setting ") +
                  Derived::class_name +
                  "'s termination: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
SK_PY_OUTCOME_TYPE::Info SK_PY_OUTCOME_CLASS::info() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return Info(this->_pyobj->attr("info"));
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name + "'s info: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting ") +
                  Derived::class_name + "'s info: " + std::string(e.what()));
    throw;
  }
}

SK_PY_OUTCOME_TEMPLATE_DECL
void SK_PY_OUTCOME_CLASS::info(const Info &i) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    this->_pyobj->attr("info") = i.pyobj();
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when setting ") +
                  Derived::class_name + "'s info: " + std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

// === TransitionOutcome implementation ===

#define SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL                                 \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_TRANSITION_OUTCOME_CLASS                                         \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::TransitionOutcome

#define SK_PY_TRANSITION_OUTCOME_TYPE                                          \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::TransitionOutcome

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::TransitionOutcome()
    : Outcome<TransitionOutcome, State>() {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::TransitionOutcome(
    std::unique_ptr<py::object> &&outcome)
    : Outcome<TransitionOutcome, State>(std::move(outcome)) {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::TransitionOutcome(const py::object &outcome)
    : Outcome<TransitionOutcome, State>(outcome) {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::TransitionOutcome(
    const State &state, const Value &transition_value,
    const Predicate &termination,
    const typename Outcome<TransitionOutcome, State>::Info &info)
    : Outcome<TransitionOutcome, State>(state, transition_value, termination,
                                        info) {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::TransitionOutcome(
    const Outcome<TransitionOutcome, State> &other)
    : Outcome<TransitionOutcome, State>(other) {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_TYPE &
SK_PY_TRANSITION_OUTCOME_CLASS::operator=(const TransitionOutcome &other) {
  static_cast<Outcome<TransitionOutcome, State> &>(*this) = other;
  return *this;
}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_CLASS::~TransitionOutcome() {}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
SK_PY_TRANSITION_OUTCOME_TYPE::State SK_PY_TRANSITION_OUTCOME_CLASS::state() {
  return this->situation();
}

SK_PY_TRANSITION_OUTCOME_TEMPLATE_DECL
void SK_PY_TRANSITION_OUTCOME_CLASS::state(const State &s) {
  this->situation(s);
}

// === EnvironmentOutcome implementation ===

#define SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL                                \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_ENVIRONMENT_OUTCOME_CLASS                                        \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::EnvironmentOutcome

#define SK_PY_ENVIRONMENT_OUTCOME_TYPE                                         \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::EnvironmentOutcome

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::EnvironmentOutcome()
    : Outcome<EnvironmentOutcome, Observation>() {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::EnvironmentOutcome(
    std::unique_ptr<py::object> &&outcome)
    : Outcome<EnvironmentOutcome, Observation>(std::move(outcome)) {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::EnvironmentOutcome(const py::object &outcome)
    : Outcome<EnvironmentOutcome, Observation>(outcome) {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::EnvironmentOutcome(
    const Observation &observation, const Value &transition_value,
    const Predicate &termination,
    const typename Outcome<EnvironmentOutcome, Observation>::Info &info)
    : Outcome<EnvironmentOutcome, Observation>(observation, transition_value,
                                               termination, info) {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::EnvironmentOutcome(
    const Outcome<EnvironmentOutcome, Observation> &other)
    : Outcome<EnvironmentOutcome, Observation>(other) {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_TYPE &
SK_PY_ENVIRONMENT_OUTCOME_CLASS::operator=(const EnvironmentOutcome &other) {
  static_cast<Outcome<EnvironmentOutcome, Observation> &>(*this) = other;
  return *this;
}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_CLASS::~EnvironmentOutcome() {}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
SK_PY_ENVIRONMENT_OUTCOME_TYPE::Observation
SK_PY_ENVIRONMENT_OUTCOME_CLASS::observation() {
  return this->situation();
}

SK_PY_ENVIRONMENT_OUTCOME_TEMPLATE_DECL
void SK_PY_ENVIRONMENT_OUTCOME_CLASS::observation(const Observation &o) {
  this->situation(o);
}

// === DistributionValue implementation ===

#define SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL                                 \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_DISTRIBUTION_VALUE_CLASS                                         \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::NextStateDistribution::DistributionValue

#define SK_PY_DISTRIBUTION_VALUE_TYPE                                          \
  typename PythonDomainProxy<                                                  \
      Texecution, Tagent, Tobservability, Tcontrollability,                    \
      Tmemory>::NextStateDistribution::DistributionValue

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
SK_PY_DISTRIBUTION_VALUE_CLASS::DistributionValue() {}

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
SK_PY_DISTRIBUTION_VALUE_CLASS::DistributionValue(const py::object &o) {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!py::isinstance<py::tuple>(o)) {
      throw std::invalid_argument(
          "SKDECIDE exception: python next state distribution returned value "
          "should be an iterable over tuple objects");
    }
    py::tuple t = o.cast<py::tuple>();
    _state = State(t[0]);
    _probability = t[1].cast<double>();
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string(
            "SKDECIDE exception when importing distribution value data: ") +
        std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
SK_PY_DISTRIBUTION_VALUE_CLASS::DistributionValue(
    const DistributionValue &other) {
  this->_state = other._state;
  this->_probability = other._probability;
}

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
SK_PY_DISTRIBUTION_VALUE_TYPE &
SK_PY_DISTRIBUTION_VALUE_CLASS::operator=(const DistributionValue &other) {
  this->_state = other._state;
  this->_probability = other._probability;
  return *this;
}

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
const SK_PY_DISTRIBUTION_VALUE_TYPE::State &
SK_PY_DISTRIBUTION_VALUE_CLASS::state() const {
  return _state;
}

SK_PY_DISTRIBUTION_VALUE_TEMPLATE_DECL
const double &SK_PY_DISTRIBUTION_VALUE_CLASS::probability() const {
  return _probability;
}

// === NextStateDistributionValues implementation ===

#define SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL                     \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS                             \
  PythonDomainProxy<                                                           \
      Texecution, Tagent, Tobservability, Tcontrollability,                    \
      Tmemory>::NextStateDistribution::NextStateDistributionValues

#define SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TYPE                              \
  typename PythonDomainProxy<                                                  \
      Texecution, Tagent, Tobservability, Tcontrollability,                    \
      Tmemory>::NextStateDistribution::NextStateDistributionValues

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::NextStateDistributionValues()
    : PyObj<NextStateDistributionValues>() {}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::NextStateDistributionValues(
    std::unique_ptr<py::object> &&next_state_distribution)
    : PyObj<NextStateDistributionValues>(std::move(next_state_distribution)) {}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::NextStateDistributionValues(
    const py::object &next_state_distribution)
    : PyObj<NextStateDistributionValues>(next_state_distribution) {}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::NextStateDistributionValues(
    const NextStateDistributionValues &other)
    : PyObj<NextStateDistributionValues>(other) {}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TYPE &
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::operator=(
    const NextStateDistributionValues &other) {
  static_cast<PyObj<NextStateDistributionValues> &>(*this) = other;
  return *this;
}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::~NextStateDistributionValues() {}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TYPE::PyIter
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::begin() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->begin());
}

SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TYPE::PyIter
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_CLASS::end() const {
  typename GilControl<Texecution>::Acquire acquire;
  return PyIter(this->_pyobj->end());
}

// === NextStateDistribution implementation ===

#define SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL                            \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_NEXT_STATE_DISTRIBUTION_CLASS                                    \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::NextStateDistribution

#define SK_PY_NEXT_STATE_DISTRIBUTION_TYPE                                     \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>::NextStateDistribution

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::NextStateDistribution()
    : PyObj<NextStateDistribution>() {
  construct();
}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::NextStateDistribution(
    std::unique_ptr<py::object> &&next_state_distribution)
    : PyObj<NextStateDistribution>(std::move(next_state_distribution)) {
  construct();
}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::NextStateDistribution(
    const py::object &next_state_distribution)
    : PyObj<NextStateDistribution>(next_state_distribution) {
  construct();
}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
void SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::construct() {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (this->_pyobj->is_none()) {
      this->_pyobj =
          std::make_unique<py::object>(skdecide::Globals::skdecide().attr(
              "DiscreteDistribution")(py::list()));
    }
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when importing next state "
                              "distribution data: ") +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::NextStateDistribution(
    const NextStateDistribution &other)
    : PyObj<NextStateDistribution>(other) {}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_TYPE &
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::operator=(
    const NextStateDistribution &other) {
  static_cast<PyObj<NextStateDistribution> &>(*this) = other;
  return *this;
}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::~NextStateDistribution() {}

SK_PY_NEXT_STATE_DISTRIBUTION_TEMPLATE_DECL
SK_PY_NEXT_STATE_DISTRIBUTION_VALUES_TYPE
SK_PY_NEXT_STATE_DISTRIBUTION_CLASS::get_values() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    if (!py::hasattr(*(this->_pyobj), "get_values")) {
      throw std::invalid_argument(
          "SKDECIDE exception: python next state distribution object must "
          "implement get_values()");
    }
    return NextStateDistributionValues(this->_pyobj->attr("get_values")());
  } catch (const py::error_already_set *e) {
    Logger::error(std::string("SKDECIDE exception when getting next state's "
                              "distribution values: ") +
                  std::string(e->what()));
    std::runtime_error err(e->what());
    delete e;
    throw err;
  } catch (const std::exception &e) {
    Logger::error(std::string("SKDECIDE exception when getting next state's "
                              "distribution values: ") +
                  std::string(e.what()));
    throw;
  }
}

// === PythonDomainProxy::Implementation<SequentialExecution> implementation ===

#define SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL                              \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>                       \
  template <typename TexecutionPolicy>

#define SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS                                      \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::                                                 \
      Implementation<TexecutionPolicy,                                         \
                     typename std::enable_if<std::is_same<                     \
                         TexecutionPolicy, SequentialExecution>::value>::type>

#define SK_PY_DOMAIN_PROXY_TYPE                                                \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::Implementation(const py::object &domain) {
  _domain = std::make_unique<py::object>(domain);
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::~Implementation() { _domain.reset(); }

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
void SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::close() {}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
std::size_t SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_parallel_capacity() {
  return 1;
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::ApplicableActionSpace
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_applicable_actions(
    const Memory &m, [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return ApplicableActionSpace(
        _domain->attr("get_applicable_actions")(m.pyobj()));
  } catch (const py::error_already_set *e) {
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
template <typename TTagent, typename TTaction, typename TagentApplicableActions>
std::enable_if_t<std::is_same<TTagent, MultiAgent>::value,
                 TagentApplicableActions>
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_agent_applicable_actions(
    const Memory &m, const TTaction &other_agents_actions, const Agent &agent,
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return TagentApplicableActions(
        _domain->attr("get_agent_applicable_actions")(
            m.pyobj(), other_agents_actions.pyobj(), agent.pyobj()));
  } catch (const py::error_already_set *e) {
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Observation SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::reset(
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return Observation(_domain->attr("reset")());
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::step(
    const Event &e, [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return EnvironmentOutcome(_domain->attr("step")(e.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::sample(
    const Memory &m, const Event &e,
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return EnvironmentOutcome(_domain->attr("sample")(m.pyobj(), e.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::State
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_next_state(
    const Memory &m, const Event &e,
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return State(_domain->attr("get_next_state")(m.pyobj(), e.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::NextStateDistribution
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_next_state_distribution(
    const Memory &m, const Event &e,
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return NextStateDistribution(
        _domain->attr("get_next_state_distribution")(m.pyobj(), e.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Value
SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::get_transition_value(
    const Memory &m, const Event &e, const State &sp,
    [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return Value(_domain->attr("get_transition_value")(m.pyobj(), e.pyobj(),
                                                       sp.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::is_goal(
    const State &s, [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return py::cast<bool>(_domain->attr("is_goal")(s.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_SEQ_IMPL_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_SEQ_IMPL_CLASS::is_terminal(
    const State &s, [[maybe_unused]] const std::size_t *thread_id) {
  try {
    return py::cast<bool>(_domain->attr("is_terminal")(s.pyobj()));
  } catch (const py::error_already_set *ex) {
    std::runtime_error err(ex->what());
    delete ex;
    throw err;
  }
}

// === PythonDomainProxy::Implementation<ParallelExecution> implementation ===

#define SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL                              \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>                       \
  template <typename TexecutionPolicy>

#define SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS                                      \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>::                                                 \
      Implementation<TexecutionPolicy,                                         \
                     typename std::enable_if<std::is_same<                     \
                         TexecutionPolicy, ParallelExecution>::value>::type>

#define SK_PY_DOMAIN_PROXY_TYPE                                                \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::Implementation(const py::object &domain) {
  typename GilControl<Texecution>::Acquire acquire;
  _domain = std::make_unique<py::object>(domain);

  if (!py::hasattr(*_domain, "get_ipc_connections")) {
    std::string err_msg =
        "SKDECIDE exception: the python domain object must provide the "
        "get_shm_files() method in parallel mode.";
    Logger::error(err_msg);
    throw std::runtime_error(err_msg);
  } else {
    try {
      py::list ipc_connections = _domain->attr("get_ipc_connections")();
      for (auto f : ipc_connections) {
        _connections.push_back(
            std::make_unique<nng::socket>(nng::pull::open()));
        _connections.back()->listen(std::string(py::str(f)).c_str());
      }
    } catch (const nng::exception &e) {
      std::string err_msg("SKDECIDE exception when trying to make pipeline "
                          "connections with the python parallel domain: ");
      err_msg += e.who() + std::string(": ") + std::string(e.what());
      Logger::error(err_msg);
      throw std::runtime_error(err_msg);
    }
  }
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::~Implementation() {
  typename GilControl<Texecution>::Acquire acquire;
  _domain.reset();
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
void SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::close() {
  try {
    _connections.clear();
  } catch (const nng::exception &e) {
    std::string err_msg("SKDECIDE exception when trying to close pipeline "
                        "connections with the python parallel domain: ");
    err_msg += e.who() + std::string(": ") + std::string(e.what());
    Logger::error(err_msg);
    throw std::runtime_error(err_msg);
  }
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
std::size_t SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_parallel_capacity() {
  typename GilControl<Texecution>::Acquire acquire;
  return py::cast<std::size_t>(_domain->attr("get_parallel_capacity")());
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::ApplicableActionSpace
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_applicable_actions(
    const Memory &m, const std::size_t *thread_id) {
  return ApplicableActionSpace(
      launch(thread_id, "get_applicable_actions", m.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
template <typename TTagent, typename TTaction, typename TagentApplicableActions>
std::enable_if_t<std::is_same<TTagent, MultiAgent>::value,
                 TagentApplicableActions>
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_agent_applicable_actions(
    const Memory &m, const TTaction &other_agents_actions, const Agent &agent,
    const std::size_t *thread_id) {
  return TagentApplicableActions(
      launch(thread_id, "get_agent_applicable_actions", m.pyobj(),
             other_agents_actions.pyobj(), agent.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Observation
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::reset(const std::size_t *thread_id) {
  return Observation(launch(thread_id, "reset"));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::step(const Event &e,
                                        const std::size_t *thread_id) {
  return EnvironmentOutcome(launch(thread_id, "step", e.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::sample(const Memory &m, const Event &e,
                                          const std::size_t *thread_id) {
  return EnvironmentOutcome(launch(thread_id, "sample", m.pyobj(), e.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::State
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_next_state(
    const Memory &m, const Event &e, const std::size_t *thread_id) {
  return State(launch(thread_id, "get_next_state", m.pyobj(), e.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::NextStateDistribution
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_next_state_distribution(
    const Memory &m, const Event &e, const std::size_t *thread_id) {
  return NextStateDistribution(
      launch(thread_id, "get_next_state_distribution", m.pyobj(), e.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Value
SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::get_transition_value(
    const Memory &m, const Event &e, const State &sp,
    const std::size_t *thread_id) {
  return Value(launch(thread_id, "get_transition_value", m.pyobj(), e.pyobj(),
                      sp.pyobj()));
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::is_goal(const State &s,
                                                const std::size_t *thread_id) {
  std::unique_ptr<py::object> r = launch(thread_id, "is_goal", s.pyobj());
  typename GilControl<Texecution>::Acquire acquire;
  try {
    bool rr = py::cast<bool>(*r);
    r.reset();
    return rr;
  } catch (const py::error_already_set *e) {
    std::runtime_error err(e->what());
    r.reset();
    delete e;
    throw err;
  }
}

SK_PY_DOMAIN_PROXY_PAR_IMPL_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_PAR_IMPL_CLASS::is_terminal(
    const State &s, const std::size_t *thread_id) {
  std::unique_ptr<py::object> r = launch(thread_id, "is_terminal", s.pyobj());
  typename GilControl<Texecution>::Acquire acquire;
  try {
    bool rr = py::cast<bool>(*r);
    r.reset();
    return rr;
  } catch (const py::error_already_set *e) {
    std::runtime_error err(e->what());
    r.reset();
    delete e;
    throw err;
  }
}

// === PythonDomainProxy implementation ===

#define SK_PY_DOMAIN_PROXY_TEMPLATE_DECL                                       \
  template <typename Texecution, typename Tagent, typename Tobservability,     \
            typename Tcontrollability, typename Tmemory>

#define SK_PY_DOMAIN_PROXY_CLASS                                               \
  PythonDomainProxy<Texecution, Tagent, Tobservability, Tcontrollability,      \
                    Tmemory>

#define SK_PY_DOMAIN_PROXY_TYPE                                                \
  typename PythonDomainProxy<Texecution, Tagent, Tobservability,               \
                             Tcontrollability, Tmemory>

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_CLASS::PythonDomainProxy(const py::object &domain) {
  _implementation = std::make_unique<Implementation<Texecution>>(domain);
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_CLASS::~PythonDomainProxy() { _implementation.reset(); }

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
void SK_PY_DOMAIN_PROXY_CLASS::close() { _implementation->close(); }

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
std::size_t SK_PY_DOMAIN_PROXY_CLASS::get_parallel_capacity() {
  return _implementation->get_parallel_capacity();
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
typename SK_PY_DOMAIN_PROXY_CLASS::ApplicableActionSpace
SK_PY_DOMAIN_PROXY_CLASS::get_applicable_actions(const Memory &m,
                                                 const std::size_t *thread_id) {
  try {
    return _implementation->get_applicable_actions(m, thread_id);
  } catch (const std::exception &e) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string("SKDECIDE exception when getting applicable actions in ") +
        Memory::AgentData::class_name + " " + m.print() + ": " +
        std::string(e.what()));
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
template <typename TTagent, typename TTaction, typename TagentApplicableActions>
std::enable_if_t<std::is_same<TTagent, MultiAgent>::value,
                 TagentApplicableActions>
SK_PY_DOMAIN_PROXY_CLASS::get_agent_applicable_actions(
    const Memory &m, const TTaction &other_agents_actions, const Agent &agent,
    const std::size_t *thread_id) {
  try {
    return _implementation->get_agent_applicable_actions(
        m, other_agents_actions, agent, thread_id);
  } catch (const std::exception &e) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string(
            "SKDECIDE exception when getting agent applicable actions in ") +
        Memory::AgentData::class_name + " " + m.print() + ": " +
        std::string(e.what()));
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Observation
SK_PY_DOMAIN_PROXY_CLASS::reset(const std::size_t *thread_id) {
  try {
    return _implementation->reset(thread_id);
  } catch (const std::exception &e) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string("SKDECIDE exception when resetting the domain: ") +
        std::string(e.what()));
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_CLASS::step(const Event &e, const std::size_t *thread_id) {
  try {
    return _implementation->step(e, thread_id);
  } catch (const std::exception &ex) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(std::string("SKDECIDE exception when stepping with action ") +
                  e.print() + ": " + ex.what());
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::EnvironmentOutcome
SK_PY_DOMAIN_PROXY_CLASS::sample(const Memory &m, const Event &e,
                                 const std::size_t *thread_id) {
  try {
    return _implementation->sample(m, e, thread_id);
  } catch (const std::exception &ex) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(std::string("SKDECIDE exception when sampling from ") +
                  Memory::AgentData::class_name + m.print() + " " +
                  " with action " + e.print() + ": " + ex.what());
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::State
SK_PY_DOMAIN_PROXY_CLASS::get_next_state(const Memory &m, const Event &e,
                                         const std::size_t *thread_id) {
  try {
    return _implementation->get_next_state(m, e, thread_id);
  } catch (const std::exception &ex) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string("SKDECIDE exception when getting next state from ") +
        Memory::AgentData::class_name + " " + m.print() +
        " and applying action " + e.print() + ": " + ex.what());
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::NextStateDistribution
SK_PY_DOMAIN_PROXY_CLASS::get_next_state_distribution(
    const Memory &m, const Event &e, const std::size_t *thread_id) {
  try {
    return _implementation->get_next_state_distribution(m, e, thread_id);
  } catch (const std::exception &ex) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string(
            "SKDECIDE exception when getting next state distribution from ") +
        Memory::AgentData::class_name + " " + m.print() +
        " and applying action " + e.print() + ": " + ex.what());
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
SK_PY_DOMAIN_PROXY_TYPE::Value
SK_PY_DOMAIN_PROXY_CLASS::get_transition_value(const Memory &m, const Event &e,
                                               const State &sp,
                                               const std::size_t *thread_id) {
  try {
    return _implementation->get_transition_value(m, e, sp, thread_id);
  } catch (const std::exception &ex) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string("SKDECIDE exception when getting value of transition (") +
        m.print() + ", " + e.print() + ") -> " + sp.print() + ": " + ex.what());
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_CLASS::is_goal(const State &s,
                                       const std::size_t *thread_id) {
  try {
    return _implementation->is_goal(s, thread_id);
  } catch (const std::exception &e) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string(
            "SKDECIDE exception when testing goal condition of state ") +
        s.print() + ": " + std::string(e.what()));
    throw;
  }
}

SK_PY_DOMAIN_PROXY_TEMPLATE_DECL
bool SK_PY_DOMAIN_PROXY_CLASS::is_terminal(const State &s,
                                           const std::size_t *thread_id) {
  try {
    return _implementation->is_terminal(s, thread_id);
  } catch (const std::exception &e) {
    typename GilControl<Texecution>::Acquire acquire;
    Logger::error(
        std::string(
            "SKDECIDE exception when testing terminal condition of state ") +
        s.print() + ": " + std::string(e.what()));
    throw;
  }
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_DOMAIN_PROXY_IMPL_HH
