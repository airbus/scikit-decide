/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_EXPRESSION_HH
#define SKDECIDE_PDDL_EXPRESSION_HH

#include <memory>
#include <ostream>
#include <sstream>

namespace skdecide {

    namespace pddl {

        class Expression {
        public :
            typedef std::shared_ptr<Expression> Ptr;

            virtual ~Expression() {}
            virtual std::ostream& print(std::ostream& o) const =0;

            std::string print() const {
                std::ostringstream o;
                print(o);
                return o.str();
            }
        };

    } // namespace pddl

} // namespace skdecide

// Expression printing operator
inline std::ostream& operator<<(std::ostream& o, const skdecide::pddl::Expression& f) {
    return f.print(o);
}

#endif // SKDECIDE_PDDL_EXPRESSION_HH