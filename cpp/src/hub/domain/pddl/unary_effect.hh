#ifndef AIRLAPS_PDDL_UNARY_EFFECT_HH
#define AIRLAPS_PDDL_UNARY_EFFECT_HH

#include "effect.hh"

namespace airlaps {

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

} // namespace airlaps

#endif // AIRLAPS_PDDL_UNARY_EFFECT_HH
