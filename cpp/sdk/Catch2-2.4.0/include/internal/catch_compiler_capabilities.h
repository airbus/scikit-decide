/*
 *  Created by Phil on 15/04/2013.
 *  Copyright 2013 Two Blue Cubes Ltd. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef TWOBLUECUBES_CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED
#define TWOBLUECUBES_CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED

// Detect a number of compiler features - by compiler
// The following features are defined:
//
// CATCH_CONFIG_COUNTER : is the __COUNTER__ macro supported?
// CATCH_CONFIG_WINDOWS_SEH : is Windows SEH supported?
// CATCH_CONFIG_POSIX_SIGNALS : are POSIX signals supported?
// CATCH_CONFIG_DISABLE_EXCEPTIONS : Are exceptions enabled?
// ****************
// Note to maintainers: if new toggles are added please document them
// in configuration.md, too
// ****************

// In general each macro has a _NO_<feature name> form
// (e.g. CATCH_CONFIG_NO_POSIX_SIGNALS) which disables the feature.
// Many features, at point of detection, define an _INTERNAL_ macro, so they
// can be combined, en-mass, with the _NO_ forms later.

#include "catch_platform.h"

#ifdef __cplusplus

#  if __cplusplus >= 201402L
#    define CATCH_CPP14_OR_GREATER
#  endif

#  if __cplusplus >= 201703L
#    define CATCH_CPP17_OR_GREATER
#  endif

#endif

#if defined(CATCH_CPP17_OR_GREATER)
#  define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif

#ifdef __clang__

#       define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
            _Pragma( "clang diagnostic push" ) \
            _Pragma( "clang diagnostic ignored \"-Wexit-time-destructors\"" ) \
            _Pragma( "clang diagnostic ignored \"-Wglobal-constructors\"")
#       define CATCH_INTERNAL_UNSUPPRESS_GLOBALS_WARNINGS \
            _Pragma( "clang diagnostic pop" )

#       define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
            _Pragma( "clang diagnostic push" ) \
            _Pragma( "clang diagnostic ignored \"-Wparentheses\"" )
#       define CATCH_INTERNAL_UNSUPPRESS_PARENTHESES_WARNINGS \
            _Pragma( "clang diagnostic pop" )

#       define CATCH_INTERNAL_SUPPRESS_UNUSED_WARNINGS \
            _Pragma( "clang diagnostic push" ) \
            _Pragma( "clang diagnostic ignored \"-Wunused-variable\"" )
#       define CATCH_INTERNAL_UNSUPPRESS_UNUSED_WARNINGS \
            _Pragma( "clang diagnostic pop" )

#endif // __clang__


////////////////////////////////////////////////////////////////////////////////
// Assume that non-Windows platforms support posix signals by default
#if !defined(CATCH_PLATFORM_WINDOWS)
    #define CATCH_INTERNAL_CONFIG_POSIX_SIGNALS
#endif

////////////////////////////////////////////////////////////////////////////////
// We know some environments not to support full POSIX signals
#if defined(__CYGWIN__) || defined(__QNX__) || defined(__EMSCRIPTEN__) || defined(__DJGPP__)
    #define CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS
#endif

#ifdef __OS400__
#       define CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS
#       define CATCH_CONFIG_COLOUR_NONE
#endif

////////////////////////////////////////////////////////////////////////////////
// Android somehow still does not support std::to_string
#if defined(__ANDROID__)
#    define CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING
#endif

////////////////////////////////////////////////////////////////////////////////
// Not all Windows environments support SEH properly
#if defined(__MINGW32__)
#    define CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH
#endif

////////////////////////////////////////////////////////////////////////////////
// PS4
#if defined(__ORBIS__)
#    define CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE
#endif

////////////////////////////////////////////////////////////////////////////////
// Cygwin
#ifdef __CYGWIN__

// Required for some versions of Cygwin to declare gettimeofday
// see: http://stackoverflow.com/questions/36901803/gettimeofday-not-declared-in-this-scope-cygwin
#   define _BSD_SOURCE

#endif // __CYGWIN__

////////////////////////////////////////////////////////////////////////////////
// Visual C++
#ifdef _MSC_VER


#  if _MSC_VER >= 1900 // Visual Studio 2015 or newer
#    define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#  endif

// Universal Windows platform does not support SEH
// Or console colours (or console at all...)
#  if defined(WINAPI_FAMILY) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
#    define CATCH_CONFIG_COLOUR_NONE
#  else
#    define CATCH_INTERNAL_CONFIG_WINDOWS_SEH
#  endif

#endif // _MSC_VER

////////////////////////////////////////////////////////////////////////////////
// Check if we are compiled with -fno-exceptions or equivalent
#if defined(__EXCEPTIONS) || defined(__cpp_exceptions) || defined(_CPPUNWIND)
#  define CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////
// DJGPP
#ifdef __DJGPP__
#  define CATCH_INTERNAL_CONFIG_NO_WCHAR
#endif // __DJGPP__

////////////////////////////////////////////////////////////////////////////////

// Use of __COUNTER__ is suppressed during code analysis in
// CLion/AppCode 2017.2.x and former, because __COUNTER__ is not properly
// handled by it.
// Otherwise all supported compilers support COUNTER macro,
// but user still might want to turn it off
#if ( !defined(__JETBRAINS_IDE__) || __JETBRAINS_IDE__ >= 20170300L )
    #define CATCH_INTERNAL_CONFIG_COUNTER
#endif

#if defined(CATCH_INTERNAL_CONFIG_COUNTER) && !defined(CATCH_CONFIG_NO_COUNTER) && !defined(CATCH_CONFIG_COUNTER)
#   define CATCH_CONFIG_COUNTER
#endif
#if defined(CATCH_INTERNAL_CONFIG_WINDOWS_SEH) && !defined(CATCH_CONFIG_NO_WINDOWS_SEH) && !defined(CATCH_CONFIG_WINDOWS_SEH) && !defined(CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH)
#   define CATCH_CONFIG_WINDOWS_SEH
#endif
// This is set by default, because we assume that unix compilers are posix-signal-compatible by default.
#if defined(CATCH_INTERNAL_CONFIG_POSIX_SIGNALS) && !defined(CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_POSIX_SIGNALS)
#   define CATCH_CONFIG_POSIX_SIGNALS
#endif
// This is set by default, because we assume that compilers with no wchar_t support are just rare exceptions.
#if !defined(CATCH_INTERNAL_CONFIG_NO_WCHAR) && !defined(CATCH_CONFIG_NO_WCHAR) && !defined(CATCH_CONFIG_WCHAR)
#   define CATCH_CONFIG_WCHAR
#endif

#if !defined(CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_CPP11_TO_STRING)
#    define CATCH_CONFIG_CPP11_TO_STRING
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS) && !defined(CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS) && !defined(CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)
#  define CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif

#if defined(CATCH_CONFIG_EXPERIMENTAL_REDIRECT)
#  define CATCH_INTERNAL_CONFIG_NEW_CAPTURE
#endif

#if defined(CATCH_INTERNAL_CONFIG_NEW_CAPTURE) && !defined(CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NEW_CAPTURE)
#  define CATCH_CONFIG_NEW_CAPTURE
#endif

#if !defined(CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED) && !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
#  define CATCH_CONFIG_DISABLE_EXCEPTIONS
#endif

#if !defined(CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS
#   define CATCH_INTERNAL_UNSUPPRESS_PARENTHESES_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS
#   define CATCH_INTERNAL_UNSUPPRESS_GLOBALS_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_UNUSED_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_UNUSED_WARNINGS
#   define CATCH_INTERNAL_UNSUPPRESS_UNUSED_WARNINGS
#endif

#if defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
#define CATCH_TRY if ((true))
#define CATCH_CATCH_ALL if ((false))
#define CATCH_CATCH_ANON(type) if ((false))
#else
#define CATCH_TRY try
#define CATCH_CATCH_ALL catch (...)
#define CATCH_CATCH_ANON(type) catch (type)
#endif


#endif // TWOBLUECUBES_CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED

