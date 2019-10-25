/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_RENDERABILITY_HH
#define AIRLAPS_RENDERABILITY_HH

#include "core.hh"

namespace airlaps {

template <typename Tstate>
class RenderableDomain : public virtual HistoryDomain<Tstate> {
public :
    virtual void render(const Memory<Tstate>& memory) = 0;

    inline void render() {
        render(_memory);
    }
};

} // namespace airlaps

#endif // AIRLAPS_RENDERABILITY_HH
