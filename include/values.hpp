//
// Created by sayan on 4/13/25.
//

#ifndef VALUES_HPP
#define VALUES_HPP
#include <ostream>

template <typename T> class Constant {
  T value;
public:
  constexpr Constant(T value) : value(value) {}
  constexpr operator T() const { return value; }
  constexpr auto eval() const { return value; }
  constexpr auto derivative() const { return Constant{T{}}; }
  friend std::ostream& operator<<(std::ostream& out, const Constant<T>& c) {
    return out << c.value;
  }
};

template <typename T> class variable {
  T value;

public:
  constexpr variable(T value) : value(value) {}
  constexpr operator T() const { return value; }
  constexpr T eval() const { return value; }
  constexpr T derivative() const { return Constant{++T{}}; }
  friend std::ostream& operator<<(std::ostream& out, const variable<T>& c) {
    return out << c.value;
  }
};

#endif // VALUES_HPP
