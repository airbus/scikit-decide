/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include <numeric>
#include "core.hh"
#include "builders/domain/memory.hh"

TEST_CASE("History Domain", "[history-domain]") {
    class TestHistoryDomain : public airlaps::HistoryDomain<int> {
    public :
        TestHistoryDomain() {
            std::vector<int> v(100);
            std::iota(v.begin(), v.end(), 1);
            this->_memory = this->_init_memory(v.begin(), v.end());
            this->_memory->push_back(this->_memory->front());
        }

        std::size_t get_memory_length() const {
            return this->_memory->size();
        }
    
    private :
        inline virtual std::size_t _get_memory_maxlen() const {
            return 50;
        }
    };

    TestHistoryDomain thd;
    REQUIRE( thd.get_memory_length() == 50 );
    REQUIRE( thd.get_last_state() == 51 );

    airlaps::Memory<std::string> m({"Hello", "I", "am", "airlaps"}, 5);
    m.push_back("of");
    m.push_back("course");
    REQUIRE( airlaps::HistoryDomain<std::string>::get_last_state(m) == "course" );
}

TEST_CASE("Finite History Domain", "[finite-history-domain]") {
    class TestFiniteHistoryDomain : public airlaps::FiniteHistoryDomain<char> {
    public :
        TestFiniteHistoryDomain() {
            std::vector<char> v(26);
            std::iota(v.begin(), v.end(), 'a');
            this->_memory = this->_init_memory(v.begin(), v.end());
        }

        const airlaps::Memory<char>& get_memory() const {
            return *(this->_memory);
        }

    private :
        inline virtual std::size_t make_memory_maxlen() {
            return 5;
        }
    };

    TestFiniteHistoryDomain tfhd;
    REQUIRE( tfhd.check_memory() == true );
    airlaps::Memory<char> m(tfhd.get_memory().begin(), tfhd.get_memory().end(), 3);
    REQUIRE( tfhd.check_memory(m) == false );
}

TEST_CASE("Markovian Domain", "[markovian-domain]") {
    class TestUninitializedMarkovianDomain : public airlaps::MarkovianDomain<int> {
    public :
        TestUninitializedMarkovianDomain() : airlaps::MarkovianDomain<int>() {}
    };

    TestUninitializedMarkovianDomain tumd;
    airlaps::Memory<int> m({1, 2, 3}, 1);
    REQUIRE( tumd.check_memory(m) == true );
    REQUIRE( tumd.get_last_state(m) == 3 );
    airlaps::Memory<int> mm({1, 2, 3}, 2);
    REQUIRE( tumd.check_memory(mm) == false );
    REQUIRE_THROWS( tumd.check_memory() );

    class TestInitializedMarkovianDomain : public airlaps::MarkovianDomain<int> {
    public :
        TestInitializedMarkovianDomain() : airlaps::MarkovianDomain<int>() {
            this->_memory = this->_init_memory({3});
        }
    };

    TestInitializedMarkovianDomain timd;
    REQUIRE_NOTHROW( timd.check_memory() );
    REQUIRE( timd.check_memory() == true );
}

TEST_CASE("Memory-Less Domain", "[memory-less-domain]") {
    airlaps::MemorylessDomain<double> mld;
    REQUIRE( mld.check_memory(airlaps::Memory<double>({1, 2}, 0)) == true );
    REQUIRE( mld.check_memory(airlaps::Memory<double>({1, 2}, 2)) == false );
    REQUIRE( mld.check_memory() == true );
    REQUIRE_THROWS_AS( mld.get_last_state(), std::out_of_range );
}
