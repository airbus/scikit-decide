/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_VALUE_HH
#define SKDECIDE_VALUE_HH

#include "core.hh"

namespace skdecide {

template <typename Tvalue>
class RewardDomain {
public :
    inline virtual bool check_value(const TransitionValue<Tvalue>& value) {
        return true;
    }
};


template <typename Tvalue>
class PositiveCostDomain : public RewardDomain<Tvalue> {
public :
    inline virtual bool check_value(const TransitionValue<Tvalue>& value) {
        return value.cost >= 0;
    }
};

} // namespace skdecide

#endif // SKDECIDE_VALUE_HH
