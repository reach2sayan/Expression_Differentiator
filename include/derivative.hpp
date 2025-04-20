//
// Created by sayan on 4/19/25.
//

#pragma once
#include "traits.hpp"

template <typename TExpression, typename... TDerivatives> class Derivative {
  std::tuple<TDerivatives...> derivatives;

  template <char... Cs>
  constexpr static auto
  make_derivative_tuples_impl(const TExpression &expression,
                              std::integer_sequence<char, Cs...>) {
    return std::tuple{make_all_constant_except<Cs>(expression).derivative()...};
  }
  template <std::size_t N>
  constexpr static auto
  make_derivative_tuples(const TExpression &expression,
                         const std::array<char, N> &labels) {
    return make_derivative_tuples_impl(expression, to_char_sequence(labels));
  }

  constexpr static std::tuple<TDerivatives...>
  make_derivatives(const TExpression &expression) {
    auto labels = collect_variable_labels(expression);
    return make_derivative_tuples(expression, labels);
  }

public:
  constexpr Derivative(const TExpression &expression)
      : derivatives{make_derivatives(expression)} {}
};
