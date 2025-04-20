//
// Created by sayan on 4/17/25.
//

#pragma once
#include "expressions.hpp"
#include "operations.hpp"
#include "traits.hpp"
#include <array>
#include <type_traits>

template <typename Tuple, typename Op, typename LHS, typename RHS,
          std::size_t... Is>
constexpr auto make_derivatives_impl(const Tuple &chars,
                                     const Expression<Op, LHS, RHS> &expr,
                                     std::index_sequence<Is...>) {
  return std::make_tuple(
      make_all_constant_except<std::tuple_element_t<Is, Tuple>::value>(expr)
          .derivative()...);
}

template <typename... Chars, typename Op, typename LHS, typename RHS>
constexpr auto make_derivatives(const std::tuple<Chars...> &chars,
                                const Expression<Op, LHS, RHS> &expr) {
  return make_derivatives_impl(chars, expr,
                               std::index_sequence_for<Chars...>{});
}

template <typename TExpression> class Equation {
private:
  TExpression expression;
  using symbolslist = typename extract_symbols_from_expr<TExpression>::type;
  using derivatives_t =
      decltype(make_derivatives(std::declval<symbolslist>(), expression));
  derivatives_t derivatives;

  constexpr static auto get_derivatives_impl(const TExpression &expr) {
    return make_derivatives(symbolslist{}, expr);
  }

public:
  using value_type = typename TExpression::value_type;
  constexpr operator value_type() const { return expression; }
  constexpr const auto &get_expression() const { return expression; }
  constexpr const auto &get_derivatives() const { return derivatives; }
  constexpr Equation(const TExpression &e)
      : expression{e}, derivatives{get_derivatives_impl(e)} {}
};

template <typename T> Equation(T &&) -> Equation<std::decay_t<T>>;
