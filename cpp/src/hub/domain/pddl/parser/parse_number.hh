/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_NUMBER_HH
#define SKDECIDE_PDDL_PARSE_NUMBER_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "numerical_expression.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct int_number
    : pegtl::sor<pegtl::one<'0'>,
                 pegtl::seq<pegtl::opt<pegtl::one<'-'>>, pegtl::range<'1', '9'>,
                            pegtl::star<pegtl::digit>>> {};

template <> struct action<int_number> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.number = std::make_shared<Number>(std::stol(in.string()));
  }
};

struct float_number
    : pegtl::seq<
          pegtl::opt<pegtl::one<'-'>>,
          pegtl::sor<pegtl::one<'0'>, pegtl::seq<pegtl::range<'1', '9'>,
                                                 pegtl::star<pegtl::digit>>>,
          pegtl::one<'.'>, pegtl::plus<pegtl::digit>> {};

template <> struct action<float_number> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.number = std::make_shared<Number>(std::stod(in.string()));
  }
};

struct number : pegtl::sor<float_number, int_number> {};

struct number_expression : number {};
template <> struct action<number_expression> {
  static void apply0(state &s) {
    s.expression = std::make_shared<NumericalExpression>(s.number);
  }
};

struct rational_number
    : pegtl::seq<
          pegtl::opt<pegtl::one<'-'>>,
          pegtl::sor<pegtl::one<'0'>, pegtl::seq<pegtl::range<'1', '9'>,
                                                 pegtl::star<pegtl::digit>>>,
          pegtl::one<'/'>, pegtl::range<'1', '9'>, pegtl::star<pegtl::digit>> {
};

template <> struct action<rational_number> {
  template <typename Input> static void apply(const Input &in, state &s) {
    std::string str = in.string();
    auto pos = str.find('/');
    double num = std::stod(str.substr(0, pos));
    double den = std::stod(str.substr(pos + 1));
    s.number = std::make_shared<Number>(num / den);
  }
};

struct probability : pegtl::sor<rational_number, float_number, int_number> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_NUMBER_HH
