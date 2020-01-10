/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2014-2019 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef _SKDECIDE_PEGTL_SPDLOG_TRACER_HH
#define _SKDECIDE_PEGTL_SPDLOG_TRACER_HH

#include <cassert>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include "pegtl/config.hpp"
#include "pegtl/normal.hpp"
#include "pegtl/internal/demangle.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace internal
      {
         template< typename Input >
         void print_current( const Input& in, std::ostringstream& o )
         {
            if( in.empty() ) {
               o << "<eof>";
            }
            else {
               const auto c = in.peek_uint8();
               switch( c ) {
                  case 0:
                     o << "<nul> = ";
                     break;
                  case 9:
                     o << "<ht> = ";
                     break;
                  case 10:
                     o << "<lf> = ";
                     break;
                  case 13:
                     o << "<cr> = ";
                     break;
                  default:
                     if( isprint( c ) ) {
                        o << '\'' << c << "' = ";
                     }
               }
               o << "(char)" << unsigned( c );
            }
         }

      }  // namespace internal

      struct trace_state
      {
         unsigned rule = 0;
         unsigned line = 0;
         std::vector< unsigned > stack;
      };

#if defined( _MSC_VER ) && ( _MSC_VER < 1910 )

      template< typename Rule >
      struct tracer
         : normal< Rule >
      {
         template< typename Input, typename... States >
         static void start( const Input& in, trace_state& ts, States&&... st )
         {
            std::ostringstream o;
            o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ++ts.rule << " ";
            o << in.position() << "  start  " << internal::demangle< Rule >() << "; current ";
            print_current( in, o );
            spdlog::debug( o.str() );
            ts.stack.push_back( ts.rule );
         }

         template< typename Input, typename... States >
         static void success( const Input& in, trace_state& ts, States&&... st )
         {
            assert( !ts.stack.empty() );
            std::ostringstream o;
            o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ts.stack.back() << " ";
            o << in.position() << " success " << internal::demangle< Rule >() << "; next ";
            print_current( in, o );
            spdlog::debug( o.str() );
            ts.stack.pop_back();
         }

         template< typename Input, typename... States >
         static void failure( const Input& in, trace_state& ts, States&&... st )
         {
            assert( !ts.stack.empty() );
            std::ostringstream o;
            o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ts.stack.back() << " ";
            o << in.position() << " failure " << internal::demangle< Rule >();
            spdlog::debug( o.str() );
            ts.stack.pop_back();
         }

         template< template< typename... > class Action, typename Iterator, typename Input, typename... States >
         static auto apply( const Iterator& begin, const Input& in, trace_state& ts, States&&... st )
            -> decltype( apply< Action >( begin, in, st... ) )
         {
            std::ostringstream o;
            o << std::setw( 6 ) << ++ts.line << "        ";
            o << in.position() << "  apply  " << internal::demangle< Rule >();
            spdlog::debug( o.str() );
            return normal< Rule >::template apply< Action >( begin, in, st... );
         }

         template< template< typename... > class Action, typename Input, typename... States >
         static auto apply0( const Input& in, trace_state& ts, States&&... st )
            -> decltype( apply0< Action >( in, st... ) )
         {
            std::ostringstream o;
            o << std::setw( 6 ) << ++ts.line << "        ";
            o << in.position() << "  apply0 " << internal::demangle< Rule >();
            spdlog::debug( o.str() );
            return normal< Rule >::template apply0< Action >( in, st... );
         }
      };

#else

      template< template< typename... > class Base >
      struct trace
      {
         template< typename Rule >
         struct control
            : Base< Rule >
         {
            template< typename Input, typename... States >
            static void start( const Input& in, trace_state& ts, States&&... st )
            {
               std::ostringstream o;
               o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ++ts.rule << " ";
               o << in.position() << "  start  " << internal::demangle< Rule >() << "; current ";
               print_current( in, o );
               spdlog::debug( o.str() );
               Base< Rule >::start( in, st... );
               ts.stack.push_back( ts.rule );
            }

            template< typename Input, typename... States >
            static void success( const Input& in, trace_state& ts, States&&... st )
            {
               assert( !ts.stack.empty() );
               std::ostringstream o;
               o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ts.stack.back() << " ";
               o << in.position() << " success " << internal::demangle< Rule >() << "; next ";
               print_current( in, o );
               spdlog::debug( o.str() );
               Base< Rule >::success( in, st... );
               ts.stack.pop_back();
            }

            template< typename Input, typename... States >
            static void failure( const Input& in, trace_state& ts, States&&... st )
            {
               assert( !ts.stack.empty() );
               std::ostringstream o;
               o << std::setw( 6 ) << ++ts.line << " " << std::setw( 6 ) << ts.stack.back() << " ";
               o << in.position() << " failure " << internal::demangle< Rule >();
               spdlog::debug( o.str() );
               Base< Rule >::failure( in, st... );
               ts.stack.pop_back();
            }

            template< template< typename... > class Action, typename Iterator, typename Input, typename... States >
            static auto apply( const Iterator& begin, const Input& in, trace_state& ts, States&&... st )
               -> decltype( Base< Rule >::template apply< Action >( begin, in, st... ) )
            {
               std::ostringstream o;
               o << std::setw( 6 ) << ++ts.line << "        ";
               o << in.position() << "  apply  " << internal::demangle< Rule >();
               spdlog::debug( o.str() );
               return Base< Rule >::template apply< Action >( begin, in, st... );
            }

            template< template< typename... > class Action, typename Input, typename... States >
            static auto apply0( const Input& in, trace_state& ts, States&&... st )
               -> decltype( Base< Rule >::template apply0< Action >( in, st... ) )
            {
               std::ostringstream o;
               o << std::setw( 6 ) << ++ts.line << "        ";
               o << in.position() << "  apply0 " << internal::demangle< Rule >();
               spdlog::debug( o.str() );
               return Base< Rule >::template apply0< Action >( in, st... );
            }
         };
      };

      template< typename Rule >
      using tracer = trace< normal >::control< Rule >;

#endif

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif // _SKDECIDE_PEGTL_SPDLOG_TRACER_HH
