/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2018-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_rule.hpp"

#include <tao/pegtl/contrib/if_then.hpp>

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         // clang-format off
         using grammar =
            if_then< one< 'a' >, one< 'b' >, one< 'c' > >::
            else_if_then< one< 'a' >, one< 'b' > >::
            else_then< one< 'c' > >;

         verify_rule< grammar >( __LINE__, __FILE__, "abc", result_type::success, 0 );
         verify_rule< grammar >( __LINE__, __FILE__, "abcd", result_type::success, 1 );
         verify_rule< grammar >( __LINE__, __FILE__, "ab", result_type::local_failure, 2 );
         verify_rule< grammar >( __LINE__, __FILE__, "c", result_type::success, 0 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
