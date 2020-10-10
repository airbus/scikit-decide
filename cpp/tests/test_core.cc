/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include "core.hh"


TEST_CASE("Implicit space", "[implicit-space]") {
    skdecide::ImplicitSpace<double> is([](const double& e) {return e > 2.0 && e < 3.0;});
    REQUIRE( is.contains(2.5) == true );
    REQUIRE( is.contains(3.5) == false );
}


TEST_CASE("Enumerable space", "[enumerable-space]") {
    struct TestEnumerableSpace : public skdecide::EnumerableSpace<char> {
        std::unordered_set<char> elements;

        TestEnumerableSpace(std::initializer_list<char> init)
        : elements(init) {}

        virtual const std::unordered_set<char>& get_elements() const {
            return elements;
        }

        virtual bool contains(const char& c) const {
            return elements.find(c) != elements.end();
        }
    };

    TestEnumerableSpace tes = {'a', 'i', 'r', 'l', 'a', 'p', 's'};
    REQUIRE ( tes.contains('l') == true );
    REQUIRE ( tes.get_elements().size() == 6 );
}


TEST_CASE("Serializable space", "[serializable-space]") {
    struct TestSerializableSpace : public skdecide::SerializableSpace<std::vector<bool>>,
                                   public std::unordered_set<std::vector<bool>> {
        TestSerializableSpace(std::initializer_list<std::vector<bool>> init)
        : std::unordered_set<std::vector<bool>>(init) {}

        virtual bool contains(const std::vector<bool>& x) const {
            return this->find(x) != this->end();
        }
    };

    TestSerializableSpace tss = {{true, true}, {false, true}, {true, false}};
    REQUIRE( tss.contains({true, false}) == true );
    REQUIRE( tss.contains({false, false}) == false );
    
    REQUIRE( tss.find(tss.to_jsonable({{false, true}, {true, false}})[0].get<std::vector<bool>>()) != tss.end() );
    REQUIRE( tss.from_jsonable({{true, true}, {false, true}, {true, false}}) == tss );
}


TEST_CASE("Implicit distribution", "[implicit-distribution]") {
    std::random_device g;
    std::uniform_real_distribution<> d(1.0, 2.0);
    skdecide::ImplicitDistribution<double> id([&d, &g]()->double {return d(g);});
    std::vector<double> v;
    std::generate_n(std::back_inserter(v), 10, [&id]() {return id.sample();});
    REQUIRE( std::all_of(v.begin(), v.end(), [](const auto& e)->bool {return e >= 1.0 && e <= 2.0;}) );
}


TEST_CASE("Discrete distribution", "[discrete-distribution]") {
    skdecide::DiscreteDistribution<std::string> dd = {{"one", 0.2}, {"two", 0.4}, {"three", 0.4}, {"two", 0.2}}; // duplicates and non-normalized weights
    std::vector<std::string> v;
    std::generate_n(std::back_inserter(v), 10, [&dd]() {return dd.sample();});
    REQUIRE( dd.get_values().at("two") == Approx(0.5) );
    REQUIRE( std::accumulate(dd.get_values().begin(), dd.get_values().end(), 0.0, [](const auto& a, const auto& e) {return a + e.second;}) == Approx(1.0) );
    REQUIRE( std::all_of(v.begin(), v.end(), [&dd](const auto& e)->bool {return dd.get_values().find(e) != dd.get_values().end();}) );
}


TEST_CASE("Single-Value distribution", "[single-value-distribution]") {
    skdecide::SingleValueDistribution<std::string> svd("unique");
    std::vector<std::string> v;
    std::generate_n(std::back_inserter(v), 10, [&svd]() {return svd.sample();});
    REQUIRE( (svd.get_values().size() == 1 && svd.get_values().at("unique") == Approx(1.0)) );
    REQUIRE( std::all_of(v.begin(), v.end(), [](const auto& e) {return e == "unique";}) );
}


TEST_CASE("Transition value", "[transition-value]") {
    skdecide::TransitionValue<skdecide::TransitionType::COST> tv(8.0);
    REQUIRE( tv.reward() + tv.cost() == Approx(0.0) );
}


TEST_CASE("Memory", "[memory]") {
    skdecide::Memory<char> m({'a', 'b', 'c', 'd', 'e'}, 3);
    REQUIRE( m == skdecide::Memory<char>({'c', 'd', 'e'}, 3) );
    m.push_back('f');
    m.push_front('a');
    REQUIRE( m == skdecide::Memory<char>({'a', 'd', 'e'}, 3) );
}
