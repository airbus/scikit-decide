/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_STRING_CONVERTER_HH
#define SKDECIDE_STRING_CONVERTER_HH

#include <sstream>

namespace skdecide {

struct StringConverter {
    template <typename T>
    static std::string from(const T& t) {
        std::ostringstream oss;
        oss << t;
        return oss.str();
    }

    template <typename T>
    static void to(const std::string& s, T& t) {
        std::istringstream iss(s);
        iss.exceptions(std::istringstream::failbit);
        iss >> t;
    }
};

} // namespace skdecide

#endif // SKDECIDE_STRING_CONVERTER_HH
