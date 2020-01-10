/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DURATION_EFFECT_HH
#define SKDECIDE_PDDL_DURATION_EFFECT_HH

#include "effect.hh"

namespace skdecide {

    namespace pddl {

        class DurativeAction;

        class DurationEffect : Effect {
        public :
            typedef std::shared_ptr<DurationEffect> Ptr;
            typedef std::shared_ptr<DurativeAction> DurativeActionPtr;

            DurationEffect() {}

            DurationEffect(const DurativeActionPtr& durative_action)
                : _durative_action(durative_action) {}
            
            DurationEffect(const DurationEffect& other)
                : _durative_action(other._durative_action) {}
            
            DurationEffect& operator= (const DurationEffect& other) {
                this->_durative_action = other._durative_action;
                return *this;
            }

            virtual ~DurationEffect() {}

            void set_durative_action(const DurativeActionPtr& durative_action) {
                _durative_action = durative_action;
            }

            const DurativeActionPtr& get_durative_action() const {
                return _durative_action;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "?duration";
                return o;
            }
        
        private :
            DurativeActionPtr _durative_action;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DURATION_EFFECT_HH