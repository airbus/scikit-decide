/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH
#define SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH

#include "effect.hh"
#include "variable_container.hh"

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class QuantifiedEffect : public Effect,
                                 public VariableContainer<Derived> {
        public :
            typedef std::shared_ptr<QuantifiedEffect<Derived>> Ptr;
            typedef typename VariableContainer<Derived>::VariablePtr VariablePtr;
            typedef typename VariableContainer<Derived>::VariableVector VariableVector;

            QuantifiedEffect() {}

            QuantifiedEffect(const Effect::Ptr& effect,
                             const VariableContainer<Derived>& variables)
                : VariableContainer<Derived>(variables),
                  _effect(effect) {}
            
            QuantifiedEffect(const QuantifiedEffect& other)
                : VariableContainer<Derived>(other),
                  _effect(other._effect) {}
            
            QuantifiedEffect& operator= (const QuantifiedEffect& other) {
                dynamic_cast<VariableContainer<Derived>&>(*this) = other;
                this->_effect = other._effect;
                return *this;
            }

            virtual ~QuantifiedEffect() {}

            QuantifiedEffect& set_effect(const Effect::Ptr& effect) {
                _effect = effect;
                return *this;
            }

            const Effect::Ptr& get_effect() const {
                return _effect;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " (";
                for (const auto& v : this->get_variables()) {
                    o << " " << *v;
                }
                o << ") " << *_effect << ")";
                return o;
            }
        
        private :
            Effect::Ptr _effect;
        };


        class UniversalEffect : public QuantifiedEffect<UniversalEffect> {
        public :
            static constexpr char class_name[] = "forall";

            typedef std::shared_ptr<UniversalEffect> Ptr;
            typedef QuantifiedEffect<UniversalEffect> VariablePtr;
            typedef QuantifiedEffect<UniversalEffect> VariableVector;

            UniversalEffect() {}

            UniversalEffect(const Effect::Ptr& effect,
                            const VariableContainer<UniversalEffect>& variables)
                : QuantifiedEffect<UniversalEffect>(effect, variables) {}
            
            UniversalEffect(const UniversalEffect& other)
                : QuantifiedEffect<UniversalEffect>(other) {}
            
            UniversalEffect& operator= (const UniversalEffect& other) {
                dynamic_cast<QuantifiedEffect<UniversalEffect>&>(*this) = other;
                return *this;
            }
        };


        class ExistentialEffect : public QuantifiedEffect<ExistentialEffect> {
        public :
            static constexpr char class_name[] = "exists";

            typedef std::shared_ptr<ExistentialEffect> Ptr;
            typedef QuantifiedEffect<ExistentialEffect> VariablePtr;
            typedef QuantifiedEffect<ExistentialEffect> VariableVector;

            ExistentialEffect() {}

            ExistentialEffect(const Effect::Ptr& effect,
                              const VariableContainer<ExistentialEffect>& variables)
                : QuantifiedEffect<ExistentialEffect>(effect, variables) {}
            
            ExistentialEffect(const ExistentialEffect& other)
                : QuantifiedEffect<ExistentialEffect>(other) {}
            
            ExistentialEffect& operator= (const ExistentialEffect& other) {
                dynamic_cast<QuantifiedEffect<ExistentialEffect>&>(*this) = other;
                return *this;
            }

            virtual ~ExistentialEffect() {}
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_QUANTIFIED_EFFECT_HH