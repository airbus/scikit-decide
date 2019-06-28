#ifndef AIRLAPS_EXECUTION_HH
#define AIRLAPS_EXECUTION_HH

#include "config.h"
#if defined(HAS_EXECUTION)
#include <execution>
#elif defined(HAS_OPENMP)
#include <omp.h>
#endif

namespace airlaps {
    
#if defined(HAS_EXECUTION)
    struct SequentialExecution {
        static constexpr std::execution::sequenced_policy policy = std::execution::seq;
        inline void protect(const std::function<void ()>& f) {
            f();
        }
        inline static std::string print() {
            return "sequential";
        }
    };

    struct ParallelExecution {
        static constexpr std::execution::parallel_policy policy = std::execution::par;
        inline void protect(const std::function<void ()>& f) {
            std::scoped_lock lock(_mutex);
            f();
        }
        inline static std::string print() {
            return "parallel (C++-17/TBB)";
        }
        std::mutex _mutex;
    };
#elif defined(HAS_OPENMP)
    struct SequentialExecution {
        static constexpr char policy = 0; // only useful for c++-17 thread support
        inline void protect(const std::function<void ()>& f) {
            f();
        }
        inline static std::string print() {
            return "sequential";
        }
    };

    struct ParallelExecution {
        static constexpr int policy = 0;
        inline void protect(const std::function<void ()>& f) {
            #pragma omp critical
            f();
        }
        inline static std::string print() {
            return "parallel (OpenMP)";
        }
    };
#else
    struct SequentialExecution {
        static constexpr char policy = 0;
        inline void protect(const std::function<void ()>& f) {
            f();
        }
        inline static std::string print() {
            return "sequential";
        }
    };

    struct ParallelExecution {
        static constexpr int policy = 0;
        inline void protect(const std::function<void ()>& f) {
            f();
        }
        inline static std::string print() {
            return "sequential (no parallelization support - compile with c++-17 parallelization feature or with openmp)";
        }
    };
#endif

} // namespace airlaps

namespace std {
#if !defined(HAS_EXECUTION)
    template <typename ExecutionPolicy, typename ForwardIt, typename UnaryFunction2>
    void for_each(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryFunction2 f) {
        for (ForwardIt i = first ; i != last ; i++) {
            f(*i);
        }
    }
#if defined(HAS_OPENMP)
    // mimic parallel std::for_each using openmp
    template <typename ForwardIt, typename UnaryFunction2>
    void for_each(int&& policy, ForwardIt first, ForwardIt last, UnaryFunction2 f) {
        #pragma omp parallel for
        for (ForwardIt i = first ; i != last ; i++) {
            f(*i);
        }
    }
#endif // defined(HAS_OPENMP)
#endif // !defined(HAS_EXECUTION)
} // namespace std

#endif // AIRLAPS_EXECUTION_HH
