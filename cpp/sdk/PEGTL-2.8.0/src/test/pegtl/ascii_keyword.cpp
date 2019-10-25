/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2017-2019 Dr. Colin Hirsch and Daniel Frey
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
         verify_analyze< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, true, false );

         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "foo", result_type::success, 0 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "foo ", result_type::success, 1 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "foo foo", result_type::success, 4 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "FOO", result_type::local_failure, 3 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "", result_type::local_failure, 0 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "f", result_type::local_failure, 1 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "fo", result_type::local_failure, 2 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, " foo", result_type::local_failure, 4 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "foo_", result_type::local_failure, 4 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "foo1", result_type::local_failure, 4 );
         verify_rule< keyword< 'f', 'o', 'o' > >( __LINE__, __FILE__, "fooa", result_type::local_failure, 4 );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
