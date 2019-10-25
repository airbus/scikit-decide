/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include "builders/domain/initialization.hh"

TEST_CASE("Initializable domain", "[initializable-domain]") {
    class TestInitializableDomain : public airlaps::InitializableDomain<float, int, std::nullptr_t> {
    public :
        virtual std::unique_ptr<airlaps::Distribution<int>> get_observation_distribution(const float& state, const std::nullptr_t& event) {
            return std::make_unique<airlaps::SingleValueDistribution<int>>(std::min(std::max(0, int(std::round(state))), 10));
        }

    private :
        virtual std::unique_ptr<airlaps::Space<int>> make_observation_space() {
            return std::make_unique<airlaps::ImplicitSpace<int>>([](const int& o)->bool{return o >=0 && o <= 10;});
        }

        virtual std::unique_ptr<airlaps::Space<float>> make_state_space() {
            return std::make_unique<airlaps::ImplicitSpace<float>>([](const float& s)->bool{return s >= -0.5 && s <= 10.5;});
        }

        virtual float _reset() {
            return 3.15;
        }
    };

    TestInitializableDomain tid;
    REQUIRE( tid.reset() == 3 );
}

TEST_CASE("UncertainInitializedDomain", "[uncertain-initialized-domain]") {
    class TestUncertainInitializedDomain : public airlaps::UncertainInitializedDomain<char, char, std::nullptr_t>,
                                           public airlaps::FullyObservableDomain<char, std::nullptr_t> {
    protected :
        virtual std::unique_ptr<airlaps::Distribution<char>> make_initial_state_distribution() {
            return std::make_unique<airlaps::DiscreteDistribution<char>>(std::initializer_list<std::pair<char, double> >{{'a', 0.5}, {'b', 0.5}});
        }
    
    private :
        virtual std::unique_ptr<airlaps::Space<char>> make_state_space() {
            return std::make_unique<airlaps::ImplicitSpace<char>>([](const char& s)->bool{return s == 'a' || s == 'b';});
        }
    };

    TestUncertainInitializedDomain tuid;
    char c = tuid.reset();
    REQUIRE( (c == 'a' || c == 'b') );
}

TEST_CASE("DeterministicInitializedDomain", "[deterministic-initialized-domain]") {
    class TestDeterministicInitializedDomain : public airlaps::DeterministicInitializedDomain<float, float, std::nullptr_t>,
                                               public airlaps::FullyObservableDomain<float, std::nullptr_t> {
    protected :
        virtual std::unique_ptr<airlaps::Space<float>> make_state_space() {
            return std::make_unique<airlaps::ImplicitSpace<float>>([](const float& s)->bool{return s >= -0.5 && s <= 10.5;});
        }

        virtual std::unique_ptr<float> make_initial_state() {
            return std::make_unique<float>(2.8f);
        }
    };

    TestDeterministicInitializedDomain tdid;
    float f = tdid.reset();
    REQUIRE( f == 2.8f );
}
