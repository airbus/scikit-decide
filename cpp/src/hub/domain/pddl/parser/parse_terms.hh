/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_TERMS_HH
#define SKDECIDE_PDDL_PARSE_TERMS_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_name.hh"

#include "variable.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// parse typed lists

template <typename PrimitiveSymbol>
struct primitive_symbols : pegtl::list<PrimitiveSymbol, ignored> {};

template <typename TypeSymbol>
struct either_primitive_types : pegtl::list<TypeSymbol, ignored> {};

template <typename TypeSymbol>
struct either_type
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<'e', 'i', 't', 'h', 'e', 'r'>, ignored>,
                     pegtl::seq<either_primitive_types<TypeSymbol>, ignored,
                                pegtl::one<')'>>> {};

struct object_type;
struct variable_type;
struct parent_type;

template <typename TypeSymbol>
struct type_spec
    : pegtl::action<
          action,
          pegtl::if_must<pegtl::seq<pegtl::one<'-'>, ignored>,
                         pegtl::sor<either_type<TypeSymbol>, TypeSymbol>>> {};

template <typename Rule> struct type_spec_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (!s.global_requirements->has_typing()) {
      throw pegtl::parse_error(
          "specifying types without enabling :typing requirement",
          in.current_position());
    }
  }
};

template <typename PrimitiveSymbol, typename TypeSymbol>
struct typed_list
    : pegtl::action<action,
                    pegtl::seq<primitive_symbols<PrimitiveSymbol>, ignored,
                               pegtl::opt<pegtl::action<
                                   type_spec_action, type_spec<TypeSymbol>>>>> {
  static_assert(std::is_same<TypeSymbol, object_type>::value ||
                    std::is_same<TypeSymbol, variable_type>::value ||
                    std::is_same<TypeSymbol, parent_type>::value,
                "unsupported typed list's type symbol");
  typedef TypeSymbol TypingSymbol;
};

template <typename Rule> struct typed_list_action {
  static void apply0(state &s) {
    typed_list_action<Rule>::template impl<typename Rule::TypingSymbol>::apply0(
        s);
  }

  template <typename TypeSymbol, typename Enable = void> struct impl;

  template <typename TypeSymbol>
  struct impl<TypeSymbol, typename std::enable_if<std::is_same<
                              TypeSymbol, object_type>::value>::type> {
    static void apply0(state &s) { s.object_list.clear(); }
  };

  template <typename TypeSymbol>
  struct impl<TypeSymbol, typename std::enable_if<std::is_same<
                              TypeSymbol, variable_type>::value>::type> {
    static void apply0(state &s) { s.variable_list.clear(); }
  };

  template <typename TypeSymbol>
  struct impl<TypeSymbol, typename std::enable_if<std::is_same<
                              TypeSymbol, parent_type>::value>::type> {
    static void apply0(state &s) { s.type_list.clear(); }
  };
};

template <typename PrimitiveSymbol, typename TypeSymbol>
struct typed_symbols
    : pegtl::star<
          pegtl::seq<pegtl::action<typed_list_action,
                                   typed_list<PrimitiveSymbol, TypeSymbol>>,
                     ignored>> {};

// parse typed object list

struct object_type : name {};

template <> struct action<object_type> {
  template <typename Input> static void apply(const Input &in, state &s) {
    try {
      s.name = in.string();
      for (const auto &o : s.object_list) {
        o->add_type(s.domain->get_type(s.name));
      }
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

template <typename ObjectSymbol>
struct typed_obj_list : typed_symbols<ObjectSymbol, object_type> {};

// parse typed variable list

struct variable_type : name {};

template <> struct action<variable_type> {
  template <typename Input> static void apply(const Input &in, state &s) {
    try {
      s.name = in.string();
      for (const auto &v : s.variable_list) {
        v->add_type(s.domain->get_type(s.name));
      }
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

template <typename VariableSymbol>
struct typed_var_list
    : typed_symbols<pegtl::seq<pegtl::one<'?'>, VariableSymbol>,
                    variable_type> {};

// parse typed type list

struct parent_type : name {};

template <> struct action<parent_type> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    Type::Ptr pt;

    try {
      pt = s.domain->get_type(s.name);
    } catch (...) {
      try {
        // declare parent type as new type of type object
        pt = s.domain->add_type(s.name);
        pt->add_type(s.domain->get_type("object"));
      } catch (const std::exception &e) {
        throw pegtl::parse_error(e.what(), in.current_position());
      }
    }

    try {
      for (const auto &t : s.type_list) {
        if (t->get_name() == "object") {
          throw pegtl::parse_error("subtyping 'object' type",
                                   in.current_position());
        }
        t->add_type(pt);
      }
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

template <typename TypeSymbol>
struct typed_type_list : typed_symbols<TypeSymbol, parent_type> {};

// parse termed symbols, i.e. a symbol name followed by optionally typed terms

template <typename Symbol, typename Term>
struct term_symbol
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored, Symbol, ignored>,
          pegtl::seq<typed_var_list<Term>, ignored, pegtl::one<')'>>> {};

template <typename Symbol, typename Term>
struct termed_symbols : pegtl::list<term_symbol<Symbol, Term>, ignored> {};

// parse parameter symbols, i.e. a symbol followed by optionally (untyped) terms

template <typename Constant, typename Variable>
struct term_list
    : pegtl::list<pegtl::sor<Constant, pegtl::seq<pegtl::one<'?'>, Variable>>,
                  ignored> {};

template <typename Symbol, typename Constant, typename Variable>
struct parameter_symbol
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored, Symbol, ignored>,
          pegtl::seq<pegtl::opt<term_list<Constant, Variable>, ignored>,
                     pegtl::one<')'>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_TERMS_HH
