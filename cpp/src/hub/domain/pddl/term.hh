/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TERM_HH
#define SKDECIDE_PDDL_TERM_HH

#include "type_container.hh"

namespace skdecide {

    namespace pddl {

        class Term {
        public :
            static constexpr char class_name[] = "term";
            virtual const std::string& get_name() const =0;
            virtual std::ostream& print(std::ostream& o) const =0;
        };

    } // namespace pddl

} // namespace skdecide

// Term printing operator
inline std::ostream& operator<<(std::ostream& o, const skdecide::pddl::Term& t) {
    return t.print(o);
}

#endif // SKDECIDE_PDDL_TERM_HH
