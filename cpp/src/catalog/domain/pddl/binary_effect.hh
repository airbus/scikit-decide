#ifndef AIRLAPS_PDDL_BINARY_EFFECT_HH
#define AIRLAPS_PDDL_BINARY_EFFECT_HH

#include "effect.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class BinaryEffect : public Effect {
        public :
            typedef std::shared_ptr<BinaryEffect<Derived>> Ptr;

            BinaryEffect() {}

            BinaryEffect(const Effect::Ptr& left_effect,
                         const Effect::Ptr& right_effect)
                : _left_effect(left_effect), _right_effect(right_effect) {}
            
            BinaryEffect(const BinaryEffect<Derived>& other)
                : _left_effect(other._left_effect),
                  _right_effect(other._right_effect) {}
            
            BinaryEffect<Derived>& operator= (const BinaryEffect<Derived>& other) {
                this->_left_effect = other._left_effect;
                this->_right_effect = other._right_effect;
                return *this;
            }

            virtual ~BinaryEffect() {}

            void set_left_effect(const Effect::Ptr& effect) {
                _left_effect = effect;
            }

            const Effect::Ptr& get_left_effect() const {
                return _left_effect;
            }

            void set_right_effect(const Effect::Ptr& effect) {
                _right_effect = effect;
            }

            const Effect::Ptr& get_right_effect() const {
                return _right_effect;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_left_effect << " " << *_right_effect << ")";
                return o;
            }

        private :
            Effect::Ptr _left_effect;
            Effect::Ptr _right_effect;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_BINARY_EFFECT_HH
