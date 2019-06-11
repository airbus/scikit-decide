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
