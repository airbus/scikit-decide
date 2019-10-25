/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER
#include "catch.hpp"

#include <tuple>

TEST_CASE( "tuple<>", "[toString][tuple]" )
{
    typedef std::tuple<> type;
    CHECK( "{ }" == ::Catch::Detail::stringify(type{}) );
    type value {};
    CHECK( "{ }" == ::Catch::Detail::stringify(value) );
}

TEST_CASE( "tuple<int>", "[toString][tuple]" )
{
    typedef std::tuple<int> type;
    CHECK( "{ 0 }" == ::Catch::Detail::stringify(type{0}) );
}


TEST_CASE( "tuple<float,int>", "[toString][tuple]" )
{
    typedef std::tuple<float,int> type;
    CHECK( "1.2f" == ::Catch::Detail::stringify(float(1.2)) );
    CHECK( "{ 1.2f, 0 }" == ::Catch::Detail::stringify(type{1.2f,0}) );
}

TEST_CASE( "tuple<string,string>", "[toString][tuple]" )
{
    typedef std::tuple<std::string,std::string> type;
    CHECK( "{ \"hello\", \"world\" }" == ::Catch::Detail::stringify(type{"hello","world"}) );
}

TEST_CASE( "tuple<tuple<int>,tuple<>,float>", "[toString][tuple]" )
{
    typedef std::tuple<std::tuple<int>,std::tuple<>,float> type;
    type value { std::tuple<int>{42}, {}, 1.2f };
    CHECK( "{ { 42 }, { }, 1.2f }" == ::Catch::Detail::stringify(value) );
}

TEST_CASE( "tuple<nullptr,int,const char *>", "[toString][tuple]" )
{
    typedef std::tuple<std::nullptr_t,int,const char *> type;
    type value { nullptr, 42, "Catch me" };
    CHECK( "{ nullptr, 42, \"Catch me\" }" == ::Catch::Detail::stringify(value) );
}

