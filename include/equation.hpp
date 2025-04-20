//
// Created by sayan on 4/17/25.
//

#pragma once
#include "expressions.hpp"
#include "operations.hpp"
#include "traits.hpp"
#include <array>
#include <type_traits>

template <typename TExpression> class Equation {
public:
  constexpr static size_t var_count = std::decay_t<TExpression>::var_count;

private:
  TExpression expression;

public:
  using value_type = typename TExpression::value_type;
  constexpr operator value_type() const { return expression; }
  constexpr const TExpression &get_expression() const { return expression; }
  constexpr Equation(const TExpression &e) : expression{e} {}
};

template <typename T> Equation(T &&) -> Equation<std::decay_t<T>>;
