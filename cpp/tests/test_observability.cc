/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include <numeric>
#include <list>
#include <set>
#include <cmath>
#include "core.hh"
#include "builders/domain/observability.hh"

TEST_CASE("Partially observable domain", "[partially-observable-domain") {
    class TestPartiallyObservableDomain : public skdecide::PartiallyObservableDomain<float, int, std::nullptr_t> {
    public :
        virtual std::unique_ptr<skdecide::Space<int>> make_observation_space() {
            return std::make_unique<skdecide::ImplicitSpace<int>>([](const int& o)->bool{return o >=0 && o <= 10;});
        }

        virtual std::unique_ptr<skdecide::Space<float>> make_state_space() {
            return std::make_unique<skdecide::ImplicitSpace<float>>([](const float& s)->bool{return s >= -0.5 && s <= 10.5;});
        }

        virtual std::unique_ptr<skdecide::Distribution<int>> get_observation_distribution(const float& state, const std::nullptr_t& event) {
            return std::make_unique<skdecide::SingleValueDistribution<int>>(std::min(std::max(0, int(std::round(state))), 10));
        }
    };

    TestPartiallyObservableDomain tpod;
    REQUIRE( tpod.is_observation(5) == true );
    REQUIRE( tpod.is_observation(-1) == false );
    REQUIRE( tpod.is_state(5.7) == true );
    REQUIRE( tpod.is_state(11) == false );
    REQUIRE( tpod.get_observation_distribution(2.8, nullptr)->sample() == 3 );
}

TEST_CASE("Fully observable domain", "[fully-observable-domain]") {
    class AlphabetSpace : public skdecide::Space<char> {
    public :
        AlphabetSpace() {
            std::vector<char> v(26);
            std::iota(v.begin(), v.end(), 'a');
            std::copy(v.begin(), v.end(), std::inserter(_alphabet, _alphabet.end()));
        }

        inline virtual bool contains(const char& x) const {
            return _alphabet.find(x) != _alphabet.end();
        }

    private :
        std::set<char> _alphabet;
    };

    class TestFullyObservableDomain : public skdecide::FullyObservableDomain<char, std::nullptr_t, AlphabetSpace> {
    private :
        virtual std::unique_ptr<AlphabetSpace> make_state_space() {
            return std::make_unique<AlphabetSpace>();
        }
    };

    TestFullyObservableDomain tfod;
    REQUIRE( tfod.get_observation_space().contains('Z') == false );
    REQUIRE( tfod.get_state_space().contains('i') == true );
    REQUIRE( tfod.get_observation_distribution('a', nullptr)->sample() == 'a' );
}
