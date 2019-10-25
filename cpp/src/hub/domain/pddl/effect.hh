/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_EFFECT_HH
#define AIRLAPS_PDDL_EFFECT_HH

#include <memory>
#include <ostream>
#include <sstream>

namespace airlaps {

    namespace pddl {

        class Effect {
        public :
            typedef std::shared_ptr<Effect> Ptr;

            virtual ~Effect() {}
            virtual std::ostream& print(std::ostream& o) const =0;

            std::string print() const {
                std::ostringstream o;
                print(o);
                return o.str();
            }
        };

    } // namespace pddl

} // namespace airlaps

// Effect printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Effect& f) {
    return f.print(o);
}

#endif // AIRLAPS_PDDL_EFFECT_HH