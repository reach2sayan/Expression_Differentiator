//
// Created by sayan on 4/17/25.
//

#pragma once
#include "expressions.hpp"
#include "operations.hpp"
#include "traits.hpp"
#include <ranges>
#include <array>
#include <unordered_map>

template <typename TExpression>
constexpr auto collect_variable_labels(const TExpression &expression) {
  constexpr std::size_t N = TExpression::var_count;
  std::array<char, N> result{};
  std::size_t index = 0;
  make_labels_array(expression, result, index);
  return result;
}

template <typename Expr, char... Cs>
constexpr auto construct_derivatives_impl(const Expr &expr,
                                           std::integer_sequence<char, Cs...>) {
  return std::array{make_all_constant_except<Cs>(expr)...};
}

template <typename TExpression>
auto construct_derivatives(const TExpression& e) {
  auto labels = collect_variable_labels(e); // std::array<char, N>
  return construct_derivatives_impl(e, std::make_integer_sequence<char, labels.size()>());
}

template <typename TExpression> class Equation {
public:
  constexpr static size_t var_count = std::decay_t<TExpression>::var_count;
private:
  TExpression expression;
  std::array<TExpression, var_count> derivatives;
public:
  using value_type = typename TExpression::value_type;

  constexpr operator value_type() const { return expression; }
  constexpr const TExpression &get_expression() const { return expression; }

  constexpr Equation(const TExpression &e)
      : expression{e}, derivatives{construct_derivatives(e)} {}
};

template <typename T> Equation(T &&) -> Equation<std::decay_t<T>>;
