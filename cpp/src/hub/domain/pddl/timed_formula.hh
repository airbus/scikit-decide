/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TIMED_FORMULA_HH
#define SKDECIDE_PDDL_TIMED_FORMULA_HH

#include "unary_formula.hh"

namespace skdecide {

    namespace pddl {

        class AtStartFormula : public UnaryFormula<AtStartFormula> {
        public :
            static constexpr char class_name[] = "at start";

            typedef std::shared_ptr<AtStartFormula> Ptr;

            AtStartFormula() {}

            AtStartFormula(const Formula::Ptr& formula)
                : UnaryFormula<AtStartFormula>(formula) {}
            
            AtStartFormula(const AtStartFormula& other)
                : UnaryFormula<AtStartFormula>(other) {}
            
            AtStartFormula& operator= (const AtStartFormula& other) {
                dynamic_cast<UnaryFormula<AtStartFormula>&>(*this) = other;
                return *this;
            }
        };


        class AtEndFormula : public UnaryFormula<AtEndFormula> {
        public :
            static constexpr char class_name[] = "at end";

            typedef std::shared_ptr<AtEndFormula> Ptr;

            AtEndFormula() {}

            AtEndFormula(const Formula::Ptr& formula)
                : UnaryFormula<AtEndFormula>(formula) {}
            
            AtEndFormula(const AtEndFormula& other)
                : UnaryFormula<AtEndFormula>(other) {}
            
            AtEndFormula& operator= (const AtEndFormula& other) {
                dynamic_cast<UnaryFormula<AtEndFormula>&>(*this) = other;
                return *this;
            }
        };


        class OverAllFormula : public UnaryFormula<OverAllFormula> {
        public :
            static constexpr char class_name[] = "over all";

            typedef std::shared_ptr<OverAllFormula> Ptr;

            OverAllFormula() {}

            OverAllFormula(const Formula::Ptr& formula)
                : UnaryFormula<OverAllFormula>(formula) {}
            
            OverAllFormula(const OverAllFormula& other)
                : UnaryFormula<OverAllFormula>(other) {}
            
            OverAllFormula& operator= (const OverAllFormula& other) {
                dynamic_cast<UnaryFormula<OverAllFormula>&>(*this) = other;
                return *this;
            }
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_TIMED_FORMULA_HH
