/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OPERATOR_CONTAINER_HH
#define SKDECIDE_PDDL_OPERATOR_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class Action;
class DurativeAction;
class Event;
class Process;

// Operator container

template <typename Derived, typename Operator>
class OperatorContainer : public AssociativeContainer<Derived, Operator> {
public:
  typedef
      typename AssociativeContainer<Derived, Operator>::SymbolPtr OperatorPtr;
  typedef
      typename AssociativeContainer<Derived, Operator>::SymbolSet OperatorSet;

  OperatorContainer(const OperatorContainer &other);
  OperatorContainer &operator=(const OperatorContainer &other);
  virtual ~OperatorContainer();

  template <typename T> const OperatorPtr &add_operator(const T &op);
  template <typename T> void remove_operator(const T &op);

  template <typename T> const OperatorPtr &get_operator(const T &op) const;

  const OperatorSet &get_operators() const;

protected:
  OperatorContainer();
};

// Action container

template <typename Derived>
class ActionContainer : public OperatorContainer<Derived, Action> {
public:
  typedef typename OperatorContainer<Derived, Action>::OperatorPtr ActionPtr;
  typedef typename OperatorContainer<Derived, Action>::OperatorSet ActionSet;

  ActionContainer(const ActionContainer &other);
  ActionContainer &operator=(const ActionContainer &other);
  virtual ~ActionContainer();

  template <typename T> const ActionPtr &add_action(const T &op);
  template <typename T> void remove_action(const T &op);

  template <typename T> const ActionPtr &get_action(const T &op) const;

  const ActionSet &get_actions() const;

protected:
  ActionContainer();
};

// Durative action container

template <typename Derived>
class DurativeActionContainer
    : public OperatorContainer<Derived, DurativeAction> {
public:
  typedef typename OperatorContainer<Derived, DurativeAction>::OperatorPtr
      DurativeActionPtr;
  typedef typename OperatorContainer<Derived, DurativeAction>::OperatorSet
      DurativeActionSet;

  DurativeActionContainer(const DurativeActionContainer &other);
  DurativeActionContainer &operator=(const DurativeActionContainer &other);
  virtual ~DurativeActionContainer();

  template <typename T>
  const DurativeActionPtr &add_durative_action(const T &op);

  template <typename T> void remove_durative_action(const T &op);

  template <typename T>
  const DurativeActionPtr &get_durative_action(const T &op) const;

  const DurativeActionSet &get_durative_actions() const;

protected:
  DurativeActionContainer();
};

// Event container

template <typename Derived>
class EventContainer : public OperatorContainer<Derived, Event> {
public:
  typedef typename OperatorContainer<Derived, Event>::OperatorPtr EventPtr;
  typedef typename OperatorContainer<Derived, Event>::OperatorSet EventSet;

  EventContainer(const EventContainer &other);
  EventContainer &operator=(const EventContainer &other);
  virtual ~EventContainer();

  template <typename T> const EventPtr &add_event(const T &op);
  template <typename T> void remove_event(const T &op);

  template <typename T> const EventPtr &get_event(const T &op) const;

  const EventSet &get_events() const;

protected:
  EventContainer();
};

// Process container

template <typename Derived>
class ProcessContainer : public OperatorContainer<Derived, Process> {
public:
  typedef typename OperatorContainer<Derived, Process>::OperatorPtr ProcessPtr;
  typedef typename OperatorContainer<Derived, Process>::OperatorSet ProcessSet;

  ProcessContainer(const ProcessContainer &other);
  ProcessContainer &operator=(const ProcessContainer &other);
  virtual ~ProcessContainer();

  template <typename T> const ProcessPtr &add_process(const T &op);
  template <typename T> void remove_process(const T &op);

  template <typename T> const ProcessPtr &get_process(const T &op) const;

  const ProcessSet &get_processes() const;

protected:
  ProcessContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/operator_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_OPERATOR_CONTAINER_HH
