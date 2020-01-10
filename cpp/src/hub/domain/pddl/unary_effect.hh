/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_UNARY_EFFECT_HH
#define SKDECIDE_PDDL_UNARY_EFFECT_HH

#include "effect.hh"

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class UnaryEffect : public Effect {
        public :
            typedef std::shared_ptr<UnaryEffect<Derived>> Ptr;

            UnaryEffect() {}

            UnaryEffect(const Effect::Ptr& effect)
                : _effect(effect) {}
            
            UnaryEffect(const UnaryEffect<Derived>& other)
                : _effect(other._effect) {}
            
            UnaryEffect<Derived>& operator= (const UnaryEffect<Derived>& other) {
                this->_effect = other._effect;
                return *this;
            }

            virtual ~UnaryEffect() {}

            void set_effect(const Effect::Ptr& effect) {
                _effect = effect;
            }

            const Effect::Ptr& get_effect() const {
                return _effect;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_effect << ")";
                return o;
            }

        protected :
            Effect::Ptr _effect;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_UNARY_EFFECT_HH
