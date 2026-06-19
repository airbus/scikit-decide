/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_INNER_SOLVER_TRAITS_HH
#define SKDECIDE_INNER_SOLVER_TRAITS_HH

#include <type_traits>

namespace skdecide {

template <typename Domain, typename = void>
struct has_get_next_state : std::false_type {};

template <typename Domain>
struct has_get_next_state<
    Domain, std::void_t<decltype(std::declval<Domain &>().get_next_state(
                std::declval<const typename Domain::State &>(),
                std::declval<const typename Domain::Action &>()))>>
    : std::true_type {};

template <typename Domain, typename = void>
struct has_get_next_state_distribution : std::false_type {};

template <typename Domain>
struct has_get_next_state_distribution<
    Domain,
    std::void_t<decltype(std::declval<Domain &>().get_next_state_distribution(
        std::declval<const typename Domain::State &>(),
        std::declval<const typename Domain::Action &>()))>> : std::true_type {};

template <typename Domain, typename = void>
struct has_is_terminal : std::false_type {};

template <typename Domain>
struct has_is_terminal<
    Domain, std::void_t<decltype(std::declval<Domain &>().is_terminal(
                std::declval<const typename Domain::State &>()))>>
    : std::true_type {};

template <typename Domain, typename = void>
struct has_is_goal : std::false_type {};

template <typename Domain>
struct has_is_goal<Domain,
                   std::void_t<decltype(std::declval<Domain &>().is_goal(
                       std::declval<const typename Domain::State &>()))>>
    : std::true_type {};

} // namespace skdecide

#endif // SKDECIDE_INNER_SOLVER_TRAITS_HH
