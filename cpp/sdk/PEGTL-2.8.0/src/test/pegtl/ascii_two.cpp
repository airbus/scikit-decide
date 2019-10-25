/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include "test.hpp"
#include "verify_analyze.hpp"
#include "verify_rule.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         verify_analyze< two< 'a' > >( __LINE__, __FILE__, true, false );

         verify_rule< two< 'a' > >( __LINE__, __FILE__, "", result_type::local_failure );
         verify_rule< two< 'a' > >( __LINE__, __FILE__, "a", result_type::local_failure );
         verify_rule< two< 'a' > >( __LINE__, __FILE__, "ab", result_type::local_failure );
         verify_rule< two< 'a' > >( __LINE__, __FILE__, "aa", result_type::success );
         verify_rule< two< 'a' > >( __LINE__, __FILE__, "aaa", result_type::success, 1 );
         verify_rule< two< 'a' > >( __LINE__, __FILE__, "aaaa", result_type::success, 2 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
