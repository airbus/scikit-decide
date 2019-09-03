#ifndef AIRLAPS_PDDL_TIMED_EFFECT_HH
#define AIRLAPS_PDDL_TIMED_EFFECT_HH

#include "unary_effect.hh"

namespace airlaps {

    namespace pddl {

        class AtStartEffect : public UnaryEffect<AtStartEffect> {
        public :
            static constexpr char class_name[] = "at start";

            typedef std::shared_ptr<AtStartEffect> Ptr;

            AtStartEffect() {}

            AtStartEffect(const Effect::Ptr& effect)
                : UnaryEffect<AtStartEffect>(effect) {}
            
            AtStartEffect(const AtStartEffect& other)
                : UnaryEffect<AtStartEffect>(other) {}
            
            AtStartEffect& operator= (const AtStartEffect& other) {
                dynamic_cast<UnaryEffect<AtStartEffect>&>(*this) = other;
                return *this;
            }
        };


        class AtEndEffect : public UnaryEffect<AtEndEffect> {
        public :
            static constexpr char class_name[] = "at end";

            typedef std::shared_ptr<AtEndEffect> Ptr;

            AtEndEffect() {}

            AtEndEffect(const Effect::Ptr& effect)
                : UnaryEffect<AtEndEffect>(effect) {}
            
            AtEndEffect(const AtEndEffect& other)
                : UnaryEffect<AtEndEffect>(other) {}
            
            AtEndEffect& operator= (const AtEndEffect& other) {
                dynamic_cast<UnaryEffect<AtEndEffect>&>(*this) = other;
                return *this;
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_TIMED_EFFECT_HH
