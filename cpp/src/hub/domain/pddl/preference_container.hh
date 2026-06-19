/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PREFERENCE_CONTAINER_HH
#define SKDECIDE_PDDL_PREFERENCE_CONTAINER_HH

#include "associative_container.hh"
#include "preference.hh"

namespace skdecide {

namespace pddl {

class Preference;

template <typename Derived>
class PreferenceContainer : public AssociativeContainer<Derived, Preference> {
public:
  typedef typename AssociativeContainer<Derived, Preference>::SymbolPtr
      PreferencePtr;
  typedef typename AssociativeContainer<Derived, Preference>::SymbolSet
      PreferenceSet;

  PreferenceContainer(const PreferenceContainer &other)
      : AssociativeContainer<Derived, Preference>(other) {}

  PreferenceContainer &operator=(const PreferenceContainer &other);

  template <typename T>
  const PreferencePtr &add_preference(const T &preference) {
    return AssociativeContainer<Derived, Preference>::add(preference);
  }

  template <typename T> void remove_preference(const T &preference) {
    AssociativeContainer<Derived, Preference>::remove(preference);
  }

  template <typename T>
  const PreferencePtr &get_preference(const T &preference) const {
    return AssociativeContainer<Derived, Preference>::get(preference);
  }

  const PreferenceSet &get_preferences() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual std::string print() const;

protected:
  PreferenceContainer() {}
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/preference_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_PREFERENCE_CONTAINER_HH
