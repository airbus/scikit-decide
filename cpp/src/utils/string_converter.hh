/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_STRING_CONVERTER_HH
#define SKDECIDE_STRING_CONVERTER_HH

#include <sstream>
#include <algorithm>

namespace skdecide {

struct StringConverter {
  template <typename T> static std::string from(const T &t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
  }

  template <typename T> static void to(const std::string &s, T &t) {
    std::istringstream iss(s);
    iss.exceptions(std::istringstream::failbit);
    iss >> t;
  }

  static std::string tolower(const std::string &s) {
    std::string ls = s;
    std::transform(ls.begin(), ls.end(), ls.begin(),
                   [](const auto &c) { return std::tolower(c); });
    return ls;
  }

  static std::string toupper(const std::string &s) {
    std::string us = s;
    std::transform(us.begin(), us.end(), us.begin(),
                   [](const auto &c) { return std::toupper(c); });
    return us;
  }
};

} // namespace skdecide

#endif // SKDECIDE_STRING_CONVERTER_HH
