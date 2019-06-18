#ifndef AIRLAPS_UTILS_HH
#define AIRLAPS_UTILS_HH

#include <type_traits>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>

#include "config.h"
#if defined(HAS_EXECUTION)
#include <execution>
#elif defined(HAS_OPENMP)
#include <omp.h>
#endif

namespace airlaps {
    
    template <typename T>
    struct has_hash {
        typedef char yes[1];
        typedef char no[2];

        template <typename C> static yes& test(typename C::Hash*);
        template <typename> static no& test(...);

        static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template <typename T>
    struct has_equal {
        typedef char yes[1];
        typedef char no[2];

        template <typename C> static yes& test(typename C::Equal*);
        template <typename> static no& test(...);

        static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template <typename T>
    struct has_less {
        typedef char yes[1];
        typedef char no[2];

        template <typename C> static yes& test(typename C::Less*);
        template <typename> static no& test(...);

        static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template <typename T>
    struct has_key {
        typedef char yes[1];
        typedef char no[2];

        template <typename C> static yes& test(typename C::Key*);
        template <typename> static no& test(...);

        static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template <typename T>
    struct Hash {
        template <typename U>
        typename std::enable_if<has_key<U>::value, std::size_t>::type
        operator()(const U& u) const {
            return typename T::Hash()(typename U::Key()(u));
        }

        template <typename U>
        typename std::enable_if<!has_key<U>::value, std::size_t>::type
        operator()(const U& u) const {
            return typename U::Hash()(u);
        }
    };

    template <typename T>
    struct Equal {
        template <typename U>
        typename std::enable_if<has_key<U>::value, bool>::type
        operator()(const U& ul, const U& ur) const {
            return typename T::Equal()(typename U::Key()(ul), typename U::Key()(ur));
        }

        template <typename U>
        typename std::enable_if<!has_key<U>::value, bool>::type
        operator()(const U& ul, const U& ur) const {
            return typename U::Equal()(ul, ur);
        }
    };

    template <typename T>
    struct Less {
        template <typename U>
        typename std::enable_if<has_key<U>::value, bool>::type
        operator()(const U& ul, const U& ur) const {
            return typename T::Less()(typename U::Key()(ul), typename U::Key()(ur));
        }

        template <typename U>
        typename std::enable_if<!has_key<U>::value, bool>::type
        operator()(const U& ul, const U& ur) const {
            return typename U::Less()(ul, ur);
        }
    };

    template <typename Key, typename RealKey = Key>
    struct SetTypeDeducer {
        typedef typename std::conditional<has_hash<RealKey>::value && has_equal<RealKey>::value,
                                        std::unordered_set<Key, Hash<RealKey>, Equal<RealKey>>,
                                        typename std::conditional<has_less<RealKey>::value,
                                                                    std::set<Key, Less<RealKey>>,
                                                                    void>::type>::type Set;
        static_assert(std::is_same<Key, RealKey>::value || has_key<Key>::value, "Key must contain a 'struct Key {...}' accessing type if Key is different from RealKey");
        static_assert(!std::is_void<Set>::value, "Key type given to SetTypeDeducer must contain either 'Hash and Equal' types or 'Less' type");
    };

    template <typename Key, typename Value, typename RealKey = Key>
    struct MapTypeDeducer {
        //static_assert(std::is_same<Key, RealKey>::value || has_key<Key>::value, "Key must contain a 'struct Key {...}' accessing type if Key is different from RealKey");
        typedef typename std::conditional<has_hash<RealKey>::value && has_equal<RealKey>::value,
                                        std::unordered_map<Key, Value, Hash<RealKey>, Equal<RealKey>>,
                                        typename std::conditional<has_less<RealKey>::value,
                                                                    std::map<Key, Value, Less<RealKey>>,
                                                                    void>::type>::type Map;
        static_assert(std::is_same<Key, RealKey>::value || has_key<Key>::value, "Key must contain a 'struct Key {...}' accessing type if Key is different from RealKey");
        static_assert(!std::is_void<Map>::value, "Key type given to MapTypeDeducer must contain either 'Hash and Equal' types or 'Less' type");
        
    };

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

#endif // AIRLAPS_UTLS_HH
