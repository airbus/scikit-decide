/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DURATION_FORMULA_HH
#define SKDECIDE_PDDL_DURATION_FORMULA_HH

#include "formula.hh"

namespace skdecide {

    namespace pddl {

        class DurativeAction;

        class DurationFormula : Formula {
        public :
            typedef std::shared_ptr<DurationFormula> Ptr;
            typedef std::shared_ptr<DurativeAction> DurativeActionPtr;

            DurationFormula() {}

            DurationFormula(const DurativeActionPtr& durative_action)
                : _durative_action(durative_action) {}
            
            DurationFormula(const DurationFormula& other)
                : _durative_action(other._durative_action) {}
            
            DurationFormula& operator= (const DurationFormula& other) {
                this->_durative_action = other._durative_action;
                return *this;
            }

            virtual ~DurationFormula() {}

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

#endif // SKDECIDE_PDDL_DURATION_FORMULA_HH