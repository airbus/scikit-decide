/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_AGGREGATION_EFFECT_HH
#define SKDECIDE_PDDL_AGGREGATION_EFFECT_HH

#include "effect.hh"
#include <vector>

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class AggregationEffect : public Effect {
        public :
            typedef std::shared_ptr<AggregationEffect<Derived>> Ptr;
            typedef Effect::Ptr EffectPtr;
            typedef std::vector<Effect::Ptr> EffectVector;

            AggregationEffect() {}
            
            AggregationEffect(const AggregationEffect& other)
                : _effects(other._effects) {}
            
            AggregationEffect& operator= (const AggregationEffect& other) {
                this->_effects = other._effects;
                return *this;
            }

            virtual ~AggregationEffect() {}

            AggregationEffect& append_effect(const Effect::Ptr& effect) {
                _effects.push_back(effect);
                return *this;
            }

            /**
             * Removes the effect at a given index.
             * Throws an exception if the given index exceeds the size of the
             * aggregation effect
             */
            AggregationEffect& remove_effect(const std::size_t& index) {
                if (index >= _effects.size()) {
                    throw std::out_of_range("SKDECIDE exception: index " + std::to_string(index) +
                                            " exceeds the size of the '" + Derived::class_name + "' effect");
                } else {
                    _effects.erase(_effects.begin() + index);
                    return *this;
                }
            }

            /**
             * Gets the effect at a given index.
             * Throws an exception if the given index exceeds the size of the
             * aggregation effect
             */
            const Effect::Ptr& effect_at(const std::size_t& index) {
                if (index >= _effects.size()) {
                    throw std::out_of_range("SKDECIDE exception: index " + std::to_string(index) +
                                            " exceeds the size of the '" + Derived::class_name + "' effect");
                } else {
                    return _effects[index];
                }
            }

            const EffectVector& get_effects() const {
                return _effects;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name;
                for (const auto& f : _effects) {
                    o << " " << *f;
                }
                o << ")";
                return o;
            }
        
        private :
            EffectVector _effects;
        };


        class ConjunctionEffect : public AggregationEffect<ConjunctionEffect> {
        public :
            static constexpr char class_name[] = "and";

            typedef std::shared_ptr<ConjunctionEffect> Ptr;
            typedef AggregationEffect<ConjunctionEffect>::EffectPtr EffectPtr;
            typedef std::vector<EffectPtr> EffectVector;

            ConjunctionEffect() {}
            
            ConjunctionEffect(const ConjunctionEffect& other)
                : AggregationEffect(other) {}
            
            ConjunctionEffect& operator= (const ConjunctionEffect& other) {
                dynamic_cast<AggregationEffect&>(*this) = other;
                return *this;
            }

            virtual ~ConjunctionEffect() {}
        };


        class DisjunctionEffect : public AggregationEffect<DisjunctionEffect> {
        public :
            static constexpr char class_name[] = "oneof";

            typedef std::shared_ptr<DisjunctionEffect> Ptr;
            typedef AggregationEffect<DisjunctionEffect>::EffectPtr EffectPtr;
            typedef std::vector<EffectPtr> EffectVector;

            DisjunctionEffect() {}
            
            DisjunctionEffect(const DisjunctionEffect& other)
                : AggregationEffect(other) {}
            
            DisjunctionEffect& operator= (const DisjunctionEffect& other) {
                dynamic_cast<AggregationEffect&>(*this) = other;
                return *this;
            }

            virtual ~DisjunctionEffect() {}
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_AGGREGATION_EFFECT_HH