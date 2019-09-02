#ifndef AIRLAPS_IW_HH
#define AIRLAPS_IW_HH

namespace airlaps {

template <typename Tdomain,
          typename Texecution_policy = ParallelExecution>
class IWSolver {
public :
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Event Action;
    typedef Texecution_policy ExecutionPolicy;

};

} // namespace airlaps

#endif // AIRLAPS_IW_HH
