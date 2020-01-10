/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MEMORY_HH
#define SKDECIDE_MEMORY_HH

#include <limits>
#include <stdexcept>

namespace skdecide {

template <typename Tstate>
class HistoryDomain {
public :
    typedef Tstate State;
    typedef Memory<Tstate> StateMemory;
    typedef std::unique_ptr<StateMemory> StateMemoryPtr;

    inline virtual bool check_memory(const StateMemory& memory) {
        return true;
    }

    inline virtual bool check_memory() {
        if (!_memory)
            throw std::invalid_argument("Uninitialized internal state memory");

        return check_memory(*_memory);
    }

    static State& get_last_state(StateMemory& memory) {
        if (memory.size() > 0) {
            return memory.back();
        } else {
            throw std::out_of_range("Attempting to get last state of empty memory object");
        }
    }

    inline State& get_last_state() {
        if (!_memory)
            throw std::invalid_argument("Uninitialized internal state memory");
        
        return get_last_state(*_memory);
    }

protected :
    StateMemoryPtr _memory;

    /**
     * Protected constructor because the class must be specialized to properly
     * initialize the state memory
     */
    HistoryDomain() {}

    template <typename InputIt>
    inline StateMemoryPtr _init_memory(InputIt iBegin, InputIt iEnd) {
        return std::make_unique<StateMemory>(iBegin, iEnd, _get_memory_maxlen());
    }

    inline StateMemoryPtr _init_memory(std::initializer_list<Tstate> iList) {
        return std::make_unique<StateMemory>(iList, _get_memory_maxlen());
    }

    inline virtual std::size_t _get_memory_maxlen() {
        return std::numeric_limits<std::size_t>::max();
    }
};


template <typename Tstate>
class FiniteHistoryDomain : public HistoryDomain<Tstate> {
public :
    typedef Tstate State;
    typedef Memory<Tstate> StateMemory;

    inline virtual bool check_memory(const StateMemory& memory) {
        return memory.maxlen() == _get_memory_maxlen();
    }

    inline virtual bool check_memory() {
        if (!(this->_memory))
            throw std::invalid_argument("Uninitialized internal state memory");

        return check_memory(*(this->_memory));
    }

protected :
    /**
     * Protected constructor because the class must be specialized to properly
     * initialize the state memory
     */
    FiniteHistoryDomain() {}

    inline virtual std::size_t _get_memory_maxlen() {
        if (!_memory_maxlen) {
            _memory_maxlen = std::make_unique<std::size_t>(make_memory_maxlen());
        }
        return *_memory_maxlen;
    }

    virtual std::size_t make_memory_maxlen() =0;

private :
    std::unique_ptr<std::size_t> _memory_maxlen;
};


template <typename Tstate>
class MarkovianDomain : public FiniteHistoryDomain<Tstate> {
public :
    typedef Tstate State;

protected :
    /**
     * Protected constructor because the class must be specialized to properly
     * initialize the state memory
     */
    MarkovianDomain() {}

    inline virtual std::size_t make_memory_maxlen() {
        return 1;
    }
};


template <typename Tstate>
class MemorylessDomain : public MarkovianDomain<Tstate> {
public :
    typedef Tstate State;

    MemorylessDomain() {
        this->_memory = this->_init_memory({});
    }
    
protected :
    inline virtual std::size_t make_memory_maxlen() {
        return 0;
    }
};

} // namespace skdecide

#endif // SKDECIDE_MEMORY_HH
