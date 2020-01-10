/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DOMAIN_HH
#define SKDECIDE_PDDL_DOMAIN_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "identifier.hh"
#include "requirements.hh"
#include "type_container.hh"
#include "object_container.hh"
#include "predicate_container.hh"
#include "function_container.hh"
#include "derived_predicate_container.hh"
#include "class_container.hh"

namespace skdecide {

    namespace pddl {
        
        class Domain : public Identifier,
                       public TypeContainer<Domain>,
                       public ObjectContainer<Domain>,
                       public PredicateContainer<Domain>,
                       public FunctionContainer<Domain>,
                       public DerivedPredicateContainer<Domain>,
                       public ClassContainer<Domain> {
        public :
            static constexpr char class_name[] = "domain";
            
            Domain();
            
            void set_name(const std::string& name);
            
            void set_requirements(const Requirements& requirements);
            const Requirements& get_requirements() const;

            std::string print() const;

            typedef TypeContainer<Domain>::TypePtr TypePtr;
            typedef TypeContainer<Domain>::TypeSet TypeSet;
            typedef ObjectContainer<Domain>::ObjectPtr ObjectPtr;
            typedef ObjectContainer<Domain>::ObjectSet ObjectSet;
            typedef PredicateContainer<Domain>::PredicatePtr PredicatePtr;
            typedef PredicateContainer<Domain>::PredicateSet PredicateSet;
            typedef DerivedPredicateContainer<Domain>::DerivedPredicatePtr DerivedPredicatePtr;
            typedef DerivedPredicateContainer<Domain>::DerivedPredicateSet DerivedPredicateSet;
            typedef FunctionContainer<Domain>::FunctionPtr FunctionPtr;
            typedef FunctionContainer<Domain>::FunctionSet FunctionSet;
            typedef ClassContainer<Domain>::ClassPtr ClassPtr;
            typedef ClassContainer<Domain>::ClassSet ClassSet;

        private :
            Requirements _requirements;
        };

    } // namespace pddl

} // namespace skdecide

// Domain printing operator
std::ostream& operator<<(std::ostream& o, const skdecide::pddl::Domain& d);

#endif // SKDECIDE_PDDL_DOMAIN_HH
