/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_EXPRESSION_HH
#define AIRLAPS_PDDL_EXPRESSION_HH

#include <memory>
#include <ostream>
#include <sstream>

namespace airlaps {

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

} // namespace airlaps

// Expression printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Expression& f) {
    return f.print(o);
}

#endif // AIRLAPS_PDDL_EXPRESSION_HH