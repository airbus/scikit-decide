/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_VALUE_HH
#define AIRLAPS_VALUE_HH

#include "core.hh"

namespace airlaps {

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

} // namespace airlaps

#endif // AIRLAPS_VALUE_HH
