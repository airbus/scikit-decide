#ifndef AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH
#define AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH

#include "binary_effect.hh"

namespace airlaps {

    namespace pddl {

        class ConditionalEffect : public Effect,
                                  public BinaryEffect {
        public :
            typedef std::shared_ptr<ConditionalEffect> Ptr;

            ConditionalEffect() {}

            ConditionalEffect(const Formula::Ptr& condition,
                              const Effect::Ptr& effect)
                : BinaryEffect(condition, effect) {}
            
            ConditionalEffect(const ConditionalEffect& other)
                : BinaryEffect(other) {}
            
            ConditionalEffect& operator= (const ConditionalEffect& other) {
                dynamic_cast<BinaryEffect&>(*this) = other;
                return *this;
            }

            virtual ~ConditionalEffect() {}

            virtual std::ostream& print(std::ostream& o) const {
                o << "(when " << *_condition << " " << *_effect << ")";
                return o;
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH
