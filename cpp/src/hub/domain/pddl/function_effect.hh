/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_FUNCTION_EFFECT_HH
#define AIRLAPS_PDDL_FUNCTION_EFFECT_HH

#include "effect.hh"
#include "function_expression.hh"

namespace airlaps {

    namespace pddl {

        class FunctionEffect : public Effect,
                               public FunctionExpression {
        public :
            typedef std::shared_ptr<FunctionEffect> Ptr;

            FunctionEffect() {}

            FunctionEffect(const Function::Ptr& function,
                           const TermContainer<FunctionExpression>& terms)
                : FunctionExpression(function, terms) {}
            
            FunctionEffect(const FunctionEffect& other)
                : FunctionExpression(other) {}
            
            FunctionEffect& operator= (const FunctionEffect& other) {
                dynamic_cast<FunctionExpression&>(*this) = other;
                return *this;
            }

            virtual ~FunctionEffect() {}

            virtual std::ostream& print(std::ostream& o) const {
                return FunctionExpression::print(o);
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_FUNCTION_EFFECT_HH