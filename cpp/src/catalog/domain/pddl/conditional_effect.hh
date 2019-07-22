#ifndef AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH
#define AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH

#include "binary_effect.hh"

namespace airlaps {

    namespace pddl {

        class ConditionalEffect : public BinaryEffect<ConditionalEffect> {
        public :
            static constexpr char class_name[] = "when";

            typedef std::shared_ptr<ConditionalEffect> Ptr;

            ConditionalEffect() {}

            ConditionalEffect(const Effect::Ptr& left_effect,
                              const Effect::Ptr& right_effect)
                : BinaryEffect<ConditionalEffect>(left_effect, right_effect) {}
            
            ConditionalEffect(const ConditionalEffect& other)
                : BinaryEffect<ConditionalEffect>(other) {}
            
            ConditionalEffect& operator= (const ConditionalEffect& other) {
                dynamic_cast<BinaryEffect<ConditionalEffect>&>(*this) = other;
                return *this;
            }

            virtual ~ConditionalEffect() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_CONDITIONAL_EFFECT_HH
