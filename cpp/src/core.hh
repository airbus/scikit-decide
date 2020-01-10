/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_CORE_HH
#define SKDECIDE_CORE_HH

#include <functional>
#include <random>
#include <vector>
#include <deque>
#include <unordered_set>
#include <unordered_map>

#include <json.hpp>
using json = nlohmann::json;

namespace skdecide {

template <typename T>
class Space {
public :
    /**
     * Type of elements of the space
     */
    typedef T element_type;

    /**
     * Default destructor
     */
    virtual ~Space() {}

    /**
     * Return boolean specifying if x is a valid member of this space
     */
    virtual bool contains(const T& x) const =0;
};


template <typename T>
class ImplicitSpace : public Space<T> {
public :
    /**
     * Type of elements of the space
     */
    typedef T element_type;

    /**
     * Constructor
     * @param containsFunctor Functor (can be a lambda expression returning
     *                        boolean specifying if x is a valid member of
     *                        this space)
     */
    ImplicitSpace(std::function<bool(const T&)> containsFunctor)
    : m_containsFunctor(containsFunctor) {}

    /**
     * Return boolean specifying if x is a valid member of this space
     */
    virtual bool contains(const T& x) const {
        return m_containsFunctor(x);
    }

private :
    std::function<bool(const T&)> m_containsFunctor;
};


template <typename T, template <typename...> class Tcontainer = std::unordered_set>
class EnumerableSpace : public Space<T> {
public :
    /**
     * Type of elements of the space
     */
    typedef T element_type;

    /**
     * Return the elements of this space
     */
    virtual const Tcontainer<T>& get_elements() const =0;
};


template <typename T>
class SamplableSpace : public Space<T> {
public :
    /**
     * Type of elements of the space
     */
    typedef T element_type;

    /**
     * Uniformly randomly sample a random element of this space
     */
    virtual T sample() const =0;
};


template <typename T, template <typename...> class Container = std::unordered_set>
class SerializableSpace : public Space<T> {
public :
    /**
     * Type of elements of the space
     */
    typedef T element_type;

    /**
     * Convert a batch of samples from this space to a JSONable data type
     */
    virtual json to_jsonable(const Container<T>& sample_n) const {
        // By default, assume identity is JSONable
        // See https://github.com/nlohmann/json#arbitrary-types-conversions
        return json(sample_n);
    }

    /**
     * Convert a JSONable data type to a batch of samples from this space
     */
    virtual Container<T> from_jsonable(const json& sample_n) const {
        // By default, assume identity is JSONable
        // See https://github.com/nlohmann/json#arbitrary-types-conversions
        return Container<T>(sample_n.get<Container<T>>());
    }
};


template <typename T>
class Distribution {
public :
    /**
     * Type of elements of the distribution
     */
    typedef T element_type;

    /**
     * Destructor
     */
    virtual ~Distribution() {}

    /**
     * Returning a sample from the distribution
     */
    virtual T sample() =0;
};


template <typename T>
class ImplicitDistribution : public Distribution<T> {
public :
    /**
     * Type of elements of the distribution
     */
    typedef T element_type;

    /**
     * Constructor
     * @param sampleFunctor Functor (can be a lambda expression) returning
     *                      a sample from the distribution
     */
    ImplicitDistribution(std::function<T()> sampleFunctor)
    : m_sampleFunctor(sampleFunctor) {}

    /**
     * Returning a sample from the distribution
     */
    virtual T sample() {
        return m_sampleFunctor();
    }

private :
    std::function<T()> m_sampleFunctor;
};


template <typename T, template <typename...> class Container = std::unordered_map,
          typename Generator = std::mt19937, typename IntType = int>
class DiscreteDistribution : public Distribution<T> {
public :
    /**
     * Type of elements of the distribution
     */
    typedef T element_type;

    /**
     * Constructor
     * @param iBegin Associative container begin iterator
     * @param iEnd Associative container end iterator
     * @param g Random number generator
     */
    template <typename InputIt>
    DiscreteDistribution(InputIt iBegin, InputIt iEnd)
    : m_generator(Generator(std::random_device()())) {
        for (InputIt i = iBegin; i != iEnd; i++) {
            std::pair<typename Container<T, double>::iterator, bool> r = m_values.insert(*i);
            if (r.second) {
                m_indexes.push_back(r.first);
            } else {
                r.first->second += i->second;
            }
        }

        if (m_values.empty()) {
            m_indexes.push_back(m_values.insert(std::make_pair(T(), 1.0)).first);
        }

        std::vector<double> probabilities;
        std::for_each(m_indexes.begin(), m_indexes.end(), [&](const auto& i){probabilities.push_back(i->second);});
        m_distribution.param(typename std::discrete_distribution<IntType>::param_type(probabilities.begin(), probabilities.end()));
        probabilities = m_distribution.probabilities(); // get normalized probabilities

        for (std::size_t i = 0 ; i < probabilities.size() ; i++) {
            m_indexes[i]->second = probabilities[i];
        }
    }

    /**
     * Constructor
     */
    DiscreteDistribution(std::initializer_list<std::pair<T, double> > iList)
    : DiscreteDistribution(iList.begin(), iList.end()) {}

    /**
     * Returning a sample from the distribution
     */
    virtual T sample() {
        return m_indexes[m_distribution(m_generator)]->first;
    }

    /**
     * Get the list of (element, probability) values
     */
    const Container<T, double>& get_values() const {
        return m_values;
    }

private :
    Container<T, double> m_values;
    std::vector<typename Container<T, double>::iterator> m_indexes;
    Generator m_generator;
    std::discrete_distribution<IntType> m_distribution;
};


template <typename T>
class SingleValueDistribution : public DiscreteDistribution<T> {
public :
    /**
     * Type of elements of the distribution
     */
    typedef T element_type;

    /**
     * Constructor
     */
    SingleValueDistribution(const T& value)
    : DiscreteDistribution<T>({{value, 1.0}}), m_value(value) {}

    /**
     * Returning a sample from the distribution
     */
    virtual T sample() {
        return m_value;
    }

    /**
     * Returning the value
     */
    const T& get_value() const {
        return m_value;
    }

private :
    T m_value;
};


enum class TransitionType {
    REWARD,
    COST
};

template <TransitionType TT = TransitionType::REWARD, typename T = double> class TransitionValue;

template <typename T>
class TransitionValue<TransitionType::REWARD, T> {
public :
    TransitionValue(const T& value) : m_value(value) {}

    inline virtual T reward() const { return m_value; }
    inline virtual T cost() const { return -m_value; }

private :
    T m_value;
};

template <typename T>
class TransitionValue<TransitionType::COST, T> {
public :
    TransitionValue(const T& value) : m_value(value) {}

    inline virtual T reward() const { return -m_value; }
    inline virtual T cost() const { return m_value; }

private :
    T m_value;
};


template <typename Tobservation, TransitionType TT, typename Tvalue, typename Tinfo = std::nullptr_t>
struct EnvironmentOutcome {
    EnvironmentOutcome(const Tobservation& observation, const Tvalue& value, bool termination, const Tinfo& info= Tinfo())
    : observation(observation), value(value), termination(termination), info(info) {}

    Tobservation observation;
    TransitionValue<TT, Tvalue> value;
    bool termination;
    Tinfo info;
};


template <typename Tstate, TransitionType TT, typename Tvalue, typename Tinfo = std::nullptr_t>
struct TransitionOutcome {
    TransitionOutcome(const Tstate& state, const Tvalue& value, bool termination, const Tinfo& info = Tinfo())
    : state(state), value(value), termination(termination), info(info) {}

    Tstate state;
    TransitionValue<TT, Tvalue> value;
    bool termination;
    Tinfo info;
};


/**
 * Deque class with maxlen feature like python deque; only for push_back and push_front!
 */
template <typename T>
class Memory : public std::deque<T> {
public :
    Memory(std::size_t maxlen = std::numeric_limits<std::size_t>::max())
    : std::deque<T>(), _maxlen(maxlen) {}
    
    template <typename InputIt>
    Memory(InputIt iBegin, InputIt iEnd, std::size_t maxlen = std::numeric_limits<std::size_t>::max())
    : std::deque<T>(iBegin, iEnd), _maxlen(maxlen) {
        if (this->size() > maxlen) {
            std::deque<T>::erase(std::deque<T>::begin(), std::deque<T>::begin() + std::deque<T>::size() - maxlen);
        }
    }

    Memory(std::initializer_list<T> iList, std::size_t maxlen = std::numeric_limits<std::size_t>::max())
    : Memory(iList.begin(), iList.end(), maxlen) {}

    Memory(const Memory& m)
    : std::deque<T>(static_cast<const std::deque<T>&>(m)), _maxlen(m._maxlen) {}

    void operator=(const Memory& m) {
        static_cast<std::deque<T>&>(*this) = static_cast<const std::deque<T>&>(m);
        _maxlen = m._maxlen;
    }

    bool operator==(const Memory& m) {
        return static_cast<const std::deque<T>&>(*this) == static_cast<const std::deque<T>&>(m)
               && _maxlen == m._maxlen;
    }

    std::size_t maxlen() const {
        return _maxlen;
    }

    void push_back(const T& value) {
        std::deque<T>::push_back(value);
        if (this->size() > _maxlen) {
            std::deque<T>::pop_front();
        }
    }

    void push_back(T&& value) {
        std::deque<T>::push_back(value);
        if (this->size() > _maxlen) {
            std::deque<T>::pop_front();
        }
    }

    void push_front(const T& value) {
        std::deque<T>::push_front(value);
        if (this->size() > _maxlen) {
            std::deque<T>::pop_back();
        }
    }

    void push_front(T&& value) {
        std::deque<T>::push_front(value);
        if (this->size() > _maxlen) {
            std::deque<T>::pop_back();
        }
    }

protected :
    std::size_t _maxlen;
};

} // namespace skdecide

#endif // SKDECIDE_CORE_HH
