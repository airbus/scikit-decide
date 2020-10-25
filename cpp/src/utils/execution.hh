/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_EXECUTION_HH
#define SKDECIDE_EXECUTION_HH

#include "config.h"
#if defined(HAS_EXECUTION)
#include <execution>
#elif defined(HAS_OPENMP)
#include <omp.h>
#endif
#include <atomic>
#include <thread>
#include <sstream>

namespace skdecide {
    
#if defined(HAS_EXECUTION)
    struct SequentialExecution {
        static constexpr std::execution::sequenced_policy policy = std::execution::seq;
        static constexpr char class_name[] = "sequential";

        struct Mutex {
            Mutex() {}
            void lock() {}
            void unlock() {}
        };

        struct RecursiveMutex {
            RecursiveMutex() {}
            void lock() {}
            void unlock() {}
        };

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, [[maybe_unused]] Tmutex& m) {
            f();
        }

        inline void protect(const std::function<void ()>& f) {
            f();
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            return std::string();
        }

        template <typename T>
        using atomic = T;
    };

    struct ParallelExecution {
        static constexpr std::execution::parallel_policy policy = std::execution::par;
        static constexpr char class_name[] = "parallel (C++-17/TBB)";

        typedef std::mutex Mutex;
        typedef std::recursive_mutex RecursiveMutex;

        Mutex _mutex;

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, Tmutex& m) {
            std::scoped_lock lock(m);
            f();
        }

        inline void protect(const std::function<void ()>& f) {
            protect(f, _mutex);
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            std::ostringstream s;
            s << " [thread " << std::this_thread::get_id() << "]";
            return s.str();
        }

        template <typename T>
        using atomic = std::atomic<T>;
    };
#elif defined(HAS_OPENMP)
    struct SequentialExecution {
        static constexpr char policy = 0; // only useful for c++-17 thread support
        static constexpr char class_name[] = "sequential";

        struct Mutex {
            Mutex() {}
            void lock() {}
            void unlock() {}
        };

        struct RecursiveMutex {
            RecursiveMutex() {}
            void lock() {}
            void unlock() {}
        };

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, [[maybe_unused]] Tmutex& m) {
            f();
        }

        inline void protect(const std::function<void ()>& f) {
            f();
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            return std::string();
        }

        template <typename T>
        using atomic = T;
    };

    struct ParallelExecution {
        static constexpr int policy = 0;
        static constexpr char class_name[] = "parallel (OpenMP)";

        struct Mutex {
            Mutex() {
                omp_init_lock(&_lock);
            }

            ~Mutex() {
                omp_destroy_lock(&_lock);
            }

            void lock() {
                omp_set_lock(&_lock);
            }

            void unlock() {
                omp_unset_lock(&_lock);
            }

            omp_lock_t _lock;
        };

        struct RecursiveMutex {
            RecursiveMutex() {
                omp_init_nest_lock(&_lock);
            }

            ~RecursiveMutex() {
                omp_destroy_nest_lock(&_lock);
            }

            void lock() {
                omp_set_nest_lock(&_lock);
            }

            void unlock() {
                omp_unset_nest_lock(&_lock);
            }

            omp_nest_lock_t _lock;
        };

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, Tmutex& m) {
            m.lock();
            f();
            m.unlock();
        }

        inline void protect(const std::function<void ()>& f) {
            #pragma omp critical
            f();
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            std::ostringstream s;
            s << " [thread " << std::this_thread::get_id() << "]";
            return s.str();
        }

        template <typename T>
        using atomic = std::atomic<T>;
    };
#else
    struct SequentialExecution {
        static constexpr char policy = 0;
        static constexpr char class_name[] = "sequential";

        struct Mutex {
            Mutex() {}
            void lock() {}
            void unlock() {}
        };

        struct RecursiveMutex {
            RecursiveMutex() {}
            void lock() {}
            void unlock() {}
        };

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, [[maybe_unused]] Tmutex& m) {
            f();
        }

        inline void protect(const std::function<void ()>& f) {
            f();
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            return std::string();
        }

        template <typename T>
        using atomic = T;
    };

    struct ParallelExecution {
        static constexpr int policy = 0;
        static constexpr char class_name[] = "sequential (no parallelization support - compile with c++-17 parallelization feature or with openmp)";

        struct Mutex {
            Mutex() {}
            void lock() {}
            void unlock() {}
        };

        struct RecursiveMutex {
            RecursiveMutex() {}
            void lock() {}
            void unlock() {}
        };

        template <typename Tmutex>
        inline void protect(const std::function<void ()>& f, [[maybe_unused]] Tmutex& m) {
            f();
        }

        inline void protect(const std::function<void ()>& f) {
            f();
        }

        inline static std::string print_type() {
            return class_name;
        }

        inline static std::string print_thread() {
            return std::string();
        }

        template <typename T>
        using atomic = T;
    };
#endif

} // namespace skdecide

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
    void for_each(int policy, ForwardIt first, ForwardIt last, UnaryFunction2 f) {
        // openMP handles random access iterators from version 3.0 only (thus not all container iterators)
        std::vector<ForwardIt> v;
        for (ForwardIt i = first ; i != last ; i++) {
            v.push_back(i);
        }
        #pragma omp parallel for
        for (int i = 0 ; i < v.size() ; i++) {
            f(*v[i]);
        }
    }
#endif // defined(HAS_OPENMP)
#endif // !defined(HAS_EXECUTION)
} // namespace std

#endif // SKDECIDE_EXECUTION_HH
