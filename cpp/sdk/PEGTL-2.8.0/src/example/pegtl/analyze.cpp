/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2017-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#include <tao/pegtl.hpp>
#include <tao/pegtl/analyze.hpp>

using namespace tao::TAO_PEGTL_NAMESPACE;  // NOLINT

struct bar;

struct foo
   : sor< digit, bar >
{
};

struct bar
   : plus< foo >
{
};

int main( int /*unused*/, char** /*unused*/ )
{
   if( analyze< foo >() != 0 ) {
      std::cout << "there are problems" << std::endl;
      return 1;
   }
   return 0;
}
