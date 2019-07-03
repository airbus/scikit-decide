#include <exception>
#include <list>

#include "pegtl.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "parser.hh"
#include "utils/pegtl_spdlog_tracer.hh"

namespace pegtl = tao::TAO_PEGTL_NAMESPACE;  // NOLINT

namespace airlaps {

    namespace pddl {

        namespace parser {

            // parsing state

            struct state {
                std::string name; // current parsed name
                Domain& domain; // PDDL domain
                Requirements requirements; // current parsed requirements
                std::list<Type::Ptr> type_list;

                state(Domain& d) : domain(d) {}
            };

            // by default a rule has no action

            template <typename Rule>
            struct action : pegtl::nothing<Rule> {};

            // Ignore spaces and comments

            struct comment : pegtl::if_must<pegtl::one<';'>, pegtl::until<pegtl::eolf>> {};
            
            struct ignored : pegtl::star<pegtl::sor<pegtl::space, comment>> {};

            // parse a name

            struct name : pegtl::seq<pegtl::plus<pegtl::identifier_first>,
                                     pegtl::star<pegtl::sor<pegtl::one<'-'>,
                                     pegtl::identifier_other>>> {};
            
            template <> struct action<name> {
                template <typename Input>
                static void apply(const Input& in, state& s) {
                    s.name = in.string();
                }
            };

            // parse domain name

            struct domain_name : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                                           pegtl::string<'d', 'o', 'm', 'a', 'i', 'n'>, ignored>,
                                                pegtl::seq<name, ignored,
                                                           pegtl::one<')'>>> {};

            template <> struct action<domain_name> {
                static void apply0(state& s) {
                    s.domain.set_name(s.name);
                }
            };

            // parse requirement keys

            struct req_equality : pegtl::string<':', 'e', 'q', 'u', 'a', 'l', 'i', 't', 'y'> {};
            
            template <> struct action<req_equality> {
                static void apply0(state& s) {
                    s.requirements.set_equality();
                }
            };

            struct req_strips : pegtl::string<':', 's', 't', 'r', 'i', 'p', 's'> {};
            
            template <> struct action<req_strips> {
                static void apply0(state& s) {
                    s.requirements.set_strips();
                }
            };

            struct req_typing : pegtl::string<':', 't', 'y', 'p', 'i', 'n', 'g'> {};
            
            template <> struct action<req_typing> {
                static void apply0(state& s) {
                    s.requirements.set_typing();
                }
            };

            struct req_negative_preconditions : pegtl::string<':', 'n', 'e', 'g', 'a', 't', 'i', 'v', 'e', '-', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_negative_preconditions> {
                static void apply0(state& s) {
                    s.requirements.set_negative_preconditions();
                }
            };

            struct req_disjunctive_preconditions : pegtl::string<':', 'd', 'i', 's', 'j', 'u', 'n', 'c', 't', 'i', 'v', 'e', '-', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_disjunctive_preconditions> {
                static void apply0(state& s) {
                    s.requirements.set_disjunctive_preconditions();
                }
            };

            struct req_existential_preconditions : pegtl::string<':', 'e', 'x', 'i', 's', 't', 'e', 'n', 't', 'i', 'a', 'l', '-', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_existential_preconditions> {
                static void apply0(state& s) {
                    s.requirements.set_existential_preconditions();
                }
            };

            struct req_universal_preconditions : pegtl::string<':', 'u', 'n', 'i', 'v', 'e', 'r', 's', 'a', 'l', '-', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_universal_preconditions> {
                static void apply0(state& s) {
                    s.requirements.set_universal_preconditions();
                }
            };

            struct req_conditional_effects : pegtl::string<':', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', '-', 'e', 'f', 'f', 'e', 'c', 't', 's'> {};
            
            template <> struct action<req_conditional_effects> {
                static void apply0(state& s) {
                    s.requirements.set_conditional_effects();
                }
            };

            struct req_fluents : pegtl::string<':', 'f', 'l', 'u', 'e', 'n', 't', 's'> {};
            
            template <> struct action<req_fluents> {
                static void apply0(state& s) {
                    s.requirements.set_fluents();
                }
            };

            struct req_durative_actions : pegtl::string<':', 'd', 'u', 'r', 'a', 't', 'i', 'v', 'e', '-', 'a', 'c', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_durative_actions> {
                static void apply0(state& s) {
                    s.requirements.set_durative_actions();
                }
            };

            struct req_time : pegtl::string<':', 't', 'i', 'm', 'e'> {};
            
            template <> struct action<req_time> {
                static void apply0(state& s) {
                    s.requirements.set_time();
                }
            };

            struct req_action_costs : pegtl::string<':', 'a', 'c', 't', 'i', 'o', 'n', '-', 'c', 'o', 's', 't', 's'> {};
            
            template <> struct action<req_action_costs> {
                static void apply0(state& s) {
                    s.requirements.set_action_costs();
                }
            };

            struct req_object_fluents : pegtl::string<':', 'o', 'b', 'j', 'e', 'c', 't', '-', 'f', 'l', 'u', 'e', 'n', 't', 's'> {};
            
            template <> struct action<req_object_fluents> {
                static void apply0(state& s) {
                    s.requirements.set_object_fluents();
                }
            };

            struct req_numeric_fluents : pegtl::string<':', 'n', 'u', 'm', 'e', 'r', 'i', 'c', '-', 'f', 'l', 'u', 'e', 'n', 't', 's'> {};
            
            template <> struct action<req_numeric_fluents> {
                static void apply0(state& s) {
                    s.requirements.set_numeric_fluents();
                }
            };

            struct req_modules : pegtl::string<':', 'm', 'o', 'd', 'u', 'l', 'e', 's'> {};
            
            template <> struct action<req_modules> {
                static void apply0(state& s) {
                    s.requirements.set_modules();
                }
            };

            struct req_adl : pegtl::string<':', 'a', 'd', 'l'> {};
            
            template <> struct action<req_adl> {
                static void apply0(state& s) {
                    s.requirements.set_adl();
                }
            };

            struct req_quantified_preconditions : pegtl::string<':', 'q', 'u', 'a', 'n', 't', 'i', 'f', 'i', 'e', 'd', '-', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};
            
            template <> struct action<req_quantified_preconditions> {
                static void apply0(state& s) {
                    s.requirements.set_quantified_preconditions();
                }
            };

            struct req_duration_inequalities : pegtl::string<':', 'd', 'u', 'r', 'a', 't', 'i', 'o', 'n', '-', 'i', 'n', 'e', 'q', 'u', 'a', 'l', 'i', 't', 'i', 'e', 's'> {};
            
            template <> struct action<req_duration_inequalities> {
                static void apply0(state& s) {
                    s.requirements.set_duration_inequalities();
                }
            };

            struct req_continuous_effects : pegtl::string<':', 'c', 'o', 'n', 't', 'i', 'n', 'u', 'o', 'u', 's', '-', 'e', 'f', 'f', 'e', 'c', 't', 's'> {};
            
            template <> struct action<req_continuous_effects> {
                static void apply0(state& s) {
                    s.requirements.set_continuous_effects();
                }
            };

            struct req_derived_predicates : pegtl::string<':', 'd', 'e', 'r', 'i', 'v', 'e', 'd', '-', 'p', 'r', 'e', 'd', 'i', 'c', 'a', 't', 'e', 's'> {};
            
            template <> struct action<req_derived_predicates> {
                static void apply0(state& s) {
                    s.requirements.set_derived_predicates();
                }
            };

            struct req_timed_initial_literals : pegtl::string<':', 't', 'i', 'm', 'e', 'd', '-', 'i', 'n', 'i', 't', 'i', 'a', 'l', '-', 'l', 'i', 't', 'e', 'r', 'a', 'l', 's'> {};
            
            template <> struct action<req_timed_initial_literals> {
                static void apply0(state& s) {
                    s.requirements.set_timed_initial_literals();
                }
            };

            struct req_preferences : pegtl::string<':', 'p', 'r', 'e', 'f', 'e', 'r', 'e', 'n', 'c', 'e', 's'> {};
            
            template <> struct action<req_preferences> {
                static void apply0(state& s) {
                    s.requirements.set_preferences();
                }
            };

            struct req_constraints : pegtl::string<':', 'c', 'o', 'n', 's', 't', 'r', 'a', 'i', 'n', 't', 's'> {};
            
            template <> struct action<req_constraints> {
                static void apply0(state& s) {
                    s.requirements.set_constraints();
                }
            };

            struct require_key : pegtl::sor<req_equality,
                                            req_strips,
                                            req_typing,
                                            req_negative_preconditions,
                                            req_disjunctive_preconditions,
                                            req_existential_preconditions,
                                            req_universal_preconditions,
                                            req_conditional_effects,
                                            req_fluents,
                                            req_durative_actions,
                                            req_action_costs,
                                            req_object_fluents,
                                            req_numeric_fluents,
                                            req_modules,
                                            req_adl,
                                            req_quantified_preconditions,
                                            req_duration_inequalities,
                                            req_continuous_effects,
                                            req_derived_predicates,
                                            req_timed_initial_literals,
                                            req_time,                       // sub-string of the previous one so must be parsed after
                                            req_preferences,
                                            req_constraints> {};


            // parse domain requirement definition

            struct domain_require_def : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                                                  pegtl::string<':', 'r', 'e', 'q', 'u', 'i', 'r', 'e', 'm', 'e', 'n', 't', 's'>, ignored>,
                                                       pegtl::seq<pegtl::list<require_key, ignored>,
                                                                  pegtl::one<')'>>> {};
            
            template <> struct action<domain_require_def> {
                static void apply0(state& s) {
                    s.domain.set_requirements(s.requirements);
                }
            };

            // parse domain typed types
            
            struct primitive_type : name {};
            template <> struct action<primitive_type> {
                template <typename Input>
                static void apply(const Input& in, state& s) {
                    s.name = in.string();
                    s.type_list.push_back(s.domain.add_type(s.name));
                }
            };

            struct primitive_types : pegtl::list<primitive_type, ignored> {};

            struct parent_type : name {};
            template <> struct action<parent_type> {
                template <typename Input>
                static void apply(const Input& in, state& s) {
                    s.name = in.string();
                    for (const auto& t : s.type_list) {
                        t->add_parent(s.domain.get_type(s.name));
                    }
                }
            };

            struct either_primitive_types : pegtl::list<parent_type, ignored> {};

            struct either_type : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                                           pegtl::string<'e', 'i', 't', 'h', 'e', 'r'>, ignored>,
                                                pegtl::seq<either_primitive_types, ignored,
                                                           pegtl::one<')'>>> {};

            struct typed_type : pegtl::seq<primitive_types, ignored,
                                           pegtl::opt<pegtl::if_must<pegtl::seq<pegtl::one<'-'>, ignored>,
                                                                     pegtl::sor<either_type,
                                                                                parent_type>>>> {};
            template<> struct action<typed_type> {
                static void apply0(state& s) {
                    s.type_list.clear();
                }
            };
            
            struct typed_types : pegtl::list<typed_type, ignored> {};

            // parse domain type names

            struct type_names : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                                          pegtl::string<':', 't', 'y', 'p', 'e', 's'>, ignored>,
                                               pegtl::seq<typed_types, ignored,
                                                          pegtl::one<')'>>> {};

            // parse domain constants

            struct domain_constants : pegtl::any {};

            // parse predicates

            struct predicates : pegtl::any {};

            // parse functionsdefinitions

            struct functions_def : pegtl::any {};

            // parse constraints definitions

            struct constraints_def : pegtl::any {};

            // parse domain classes

            struct classes : pegtl::any {};

            // parse structures definitions

            struct structure_defs : pegtl::any {};

            // parse domain preamble

            struct preamble_item : pegtl::sor<domain_require_def,
                                              type_names,
                                              domain_constants,
                                              predicates,
                                              functions_def,
                                              constraints_def,
                                              classes,
                                              structure_defs> {};

            struct preamble : pegtl::list<preamble_item, ignored>{};

            // parse domain

            struct domain : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                                      pegtl::string<'d', 'e', 'f', 'i', 'n', 'e'>, ignored,
                                                      domain_name, ignored>,
                                           pegtl::seq<preamble, ignored,
                                                      pegtl::one<')'>>> {};

            struct problem : pegtl::any {};

            struct plan : pegtl::any {};

            struct grammar : pegtl::sor<domain, problem, plan> {};

        } // namespace parser

        void Parser::parse(const std::string& domain_file, const std::string& problem_file,
                           Domain& domain,
                           bool debug_logs) {
            std::string parsed_files_msg = domain_file + (problem_file.empty() ? "" : (" and " + problem_file));
            
            if (debug_logs) {
                spdlog::set_level(spdlog::level::debug);
            } else {
                spdlog::set_level(spdlog::level::info);
            }

            try {
                // Parse the domain
                spdlog::info("Parsing " + parsed_files_msg);
                parser::state s(domain);
                pegtl::read_input<> in(domain_file);

                if (debug_logs) {
                    pegtl::trace_state t;
                    if (!pegtl::parse<parser::grammar, parser::action, pegtl::tracer>(in, t, s)) {
                        spdlog::error("unable to parse " + parsed_files_msg);
                        throw std::runtime_error("AIRLAPS exception: unable to parse " + parsed_files_msg);
                    }
                } else {
                    if (!pegtl::parse<parser::grammar, parser::action>(in, s)) {
                        spdlog::error("unable to parse " + parsed_files_msg);
                        throw std::runtime_error("AIRLAPS exception: unable to parse " + parsed_files_msg);
                    }
                }
            } catch (const pegtl::parse_error& e) {
                spdlog::error("Error when parsing " + parsed_files_msg + ": " + e.what());
                throw std::runtime_error("AIRLAPS exception: error when parsing " + parsed_files_msg + ": " + e.what());
            } catch (const pegtl::input_error& e) {
                spdlog::error("Error when reading " + parsed_files_msg + ": " + e.what());
                throw std::runtime_error("AIRLAPS exception: error when reading " + parsed_files_msg + ": " + e.what());
            } catch (const std::exception& e) {
                throw;
            }
        }

    } // namespace pddl

} // namespace airlaps
