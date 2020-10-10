/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include <vector>
#include <set>
#include "builders/domain/goals.hh"

TEST_CASE("GoalDomain", "[goal-domain]") {
    class TestGoalDomain : public skdecide::GoalDomain<int> {
    private :
        virtual std::unique_ptr<skdecide::Space<int>> make_goals() {
            return std::make_unique<skdecide::ImplicitSpace<int>>([](const int& s)->bool{return s >= 'h' && s <= 'm';});
        }
    };

    TestGoalDomain tgd;
    std::vector<char> v(26);
    std::iota(v.begin(), v.end(), 'a');
    std::set<char> alphabet;
    std::copy(v.begin(), v.end(), std::inserter(alphabet, alphabet.end()));
    std::set<char> goals;
    for (const auto& c : alphabet) {
        if (tgd.get_goals().contains(c)) {
            goals.insert(c);
        }
    }
    std::vector<char> vv(6);
    std::iota(vv.begin(), vv.end(), 'h');
    std::set<char> gg;
    std::copy(vv.begin(), vv.end(), std::inserter(gg, gg.end()));
    REQUIRE( goals == gg );
}
