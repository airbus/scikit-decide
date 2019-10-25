/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2017-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include <cstring>

#include "test.hpp"

#include <tao/pegtl.hpp>

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      void unit_test()
      {
         char data[ 12 ];  // NOLINT
         std::memcpy( data, "foo\0bar\0baz", 12 );
         char* argv[] = { data, data + 4, data + 8 };  // NOLINT
         argv_input<> in( argv, 1 );
         TAO_PEGTL_TEST_ASSERT( in.source() == "argv[1]" );
         const auto result = parse< string< 'b', 'a', 'r' > >( in );
         TAO_PEGTL_TEST_ASSERT( result );
      }

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#include "main.hpp"
