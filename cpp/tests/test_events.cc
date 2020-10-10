/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include <set>
#include "builders/domain/events.hh"

TEST_CASE("Event domain", "[event-domain]") {
    class AlphabetSpace : public skdecide::Space<char> {
    public :
        AlphabetSpace(bool with_capitals) {
            std::vector<char> v(26);
            std::iota(v.begin(), v.end(), 'a');
            std::copy(v.begin(), v.end(), std::inserter(_alphabet, _alphabet.end()));
            if (with_capitals) {
                std::iota(v.begin(), v.end(), 'A');
                std::copy(v.begin(), v.end(), std::inserter(_alphabet, _alphabet.end()));
            }
        }

        inline virtual bool contains(const char& x) const {
            return _alphabet.find(x) != _alphabet.end();
        }

    private :
        std::set<char> _alphabet;
    };
    
    class TestEventDomain : public skdecide::EventDomain<int, char,
                                                        AlphabetSpace, AlphabetSpace,
                                                        skdecide::ImplicitSpace<char>, skdecide::ImplicitSpace<char>> {
    public :
        virtual std::unique_ptr<skdecide::ImplicitSpace<char>> get_enabled_events(const skdecide::Memory<int>& memory) {
            return std::make_unique<skdecide::ImplicitSpace<char>>([this, &memory](const char& c)->bool {
                return (std::find(memory.begin(), memory.end(), int(c)) != memory.end()) &&
                       (this->get_event_space().contains(c));
            });
        }

        virtual std::unique_ptr<skdecide::ImplicitSpace<char>> get_applicable_actions(const skdecide::Memory<int>& memory) {
            return std::make_unique<skdecide::ImplicitSpace<char>>([this, &memory](const char& c)->bool {
                return (std::find(memory.begin(), memory.end(), int(c)) != memory.end()) &&
                       (this->get_action_space().contains(c));
            });
        }
    
    private :
        inline virtual std::unique_ptr<AlphabetSpace> make_event_space() {
            return std::make_unique<AlphabetSpace>(true);
        }

        inline virtual std::unique_ptr<AlphabetSpace> make_action_space() {
            return std::make_unique<AlphabetSpace>(false);
        }
    };

    TestEventDomain ted;
    std::vector<char> v(26);
    std::iota(v.begin(), v.end(), 'a');
    std::iota(v.begin(), v.end(), 'A');
    bool test1 = true;
    bool test2 = true;
    for (char c : v) {
        test1 = test1 && (!ted.get_action_space().contains(c) || ted.get_event_space().contains(c));
        test2 = test2 && ted.get_event_space().contains(c);
    }
    REQUIRE( test1 == true );
    REQUIRE( test2 == true );
    REQUIRE( ted.get_enabled_events({int('t'), int('@'), int('R')})->contains('t') == true );
    REQUIRE( ted.get_enabled_events({int('t'), int('@'), int('R')})->contains('@') == false );
    REQUIRE( ted.get_enabled_events({int('t'), int('@'), int('R')})->contains('R') == true );
    REQUIRE( ted.get_applicable_actions({int('t'), int('@'), int('R')})->contains('t') == true );
    REQUIRE( ted.get_applicable_actions({int('t'), int('@'), int('R')})->contains('@') == false );
    REQUIRE( ted.get_applicable_actions({int('t'), int('@'), int('R')})->contains('R') == false );
}
