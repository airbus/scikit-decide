/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_NUMBER_HH
#define SKDECIDE_PDDL_NUMBER_HH

#include <memory>

namespace skdecide {

namespace pddl {

class Number {
public:
  typedef std::shared_ptr<Number> Ptr;

  template <typename T> Number(const T &n);

  bool is_double() const;
  double as_double() const;
  long as_long() const;
  std::ostream &print(std::ostream &o) const;

private:
  struct ImplBase {
    virtual ~ImplBase();
    virtual bool is_double() const = 0;
    virtual double as_double() const = 0;
    virtual long as_long() const = 0;
    virtual std::ostream &print(std::ostream &o) const = 0;
  };

  template <typename T> struct Impl : public ImplBase {
    T _n;
    Impl(const T &n);
    virtual ~Impl();
    virtual bool is_double() const;
    virtual double as_double() const;
    virtual long as_long() const;
    virtual std::ostream &print(std::ostream &o) const;
  };

  std::unique_ptr<ImplBase> _impl;
};

// Number printing operator
std::ostream &operator<<(std::ostream &o, const Number &n);

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/number_impl.hh"
#endif

#endif // SKDECIDE_PDDL_NUMBER_HH
