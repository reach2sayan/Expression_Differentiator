//
// Created by sayan on 4/18/25.
//

#pragma once
#include "expressions.hpp"
#include "values.hpp"
#include <array>
#include <type_traits>

template <typename T> constexpr static bool is_const = false;
template <typename T> constexpr static bool is_const<Constant<T>> = true;

template <typename T, char symbol = 'X'> struct make_all_constant {
  using type = T;
};

template <typename T, char symbol>
struct make_all_constant<Variable<T, symbol>> {
  using type = Constant<T>;
};

template <typename Op, typename... TExpressions>
struct make_all_constant<Expression<Op, TExpressions...>> {
  using type =
      Expression<Op, typename make_all_constant<TExpressions>::type...>;
};

template <typename TExpression>
using as_const_expression = make_all_constant<
    Expression<typename TExpression::op_type, typename TExpression::lhs_type,
               typename TExpression::rhs_type>>::type;

template <char symbol, typename Expr> struct replace_matching_variable_as_const;
template <char symbol, typename T> struct replace_matching_variable_as_const {
  using type = T;
};

template <char symbol, typename T>
struct replace_matching_variable_as_const<symbol, Variable<T, symbol>> {
  using type = Constant<T>;
};

template <char symbol, typename Op, typename... TExpressions>
struct replace_matching_variable_as_const<symbol,
                                          Expression<Op, TExpressions...>> {
  using type = Expression<Op, typename replace_matching_variable_as_const<
                                  symbol, TExpressions>::type...>;
};

template <char symbol, typename TExpression>
using replace_matching_variable_as_const_t =
    typename replace_matching_variable_as_const<symbol, TExpression>::type;

template <char symbol, typename T>
constexpr auto make_const_variable(const Variable<T, symbol> &var) {
  return Constant<T>(var);
}

template <char symbol, typename T, char othersymbol>
constexpr auto make_const_variable(const Variable<T, othersymbol> &var)
    -> std::enable_if_t<(symbol != othersymbol),
                        Variable<T, othersymbol>> { // without the enable_if_t
                                                    // it is ambiguous for two
                                                    // variables with same label
  return var;
}

template <char symbol, typename T>
constexpr auto make_const_variable(const Constant<T> &c) {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_const_variable(const Expression<Op, LHS, RHS> &expr)
    -> Expression<Op, replace_matching_variable_as_const_t<symbol, LHS>,
                  replace_matching_variable_as_const_t<symbol, RHS>> {
  auto lexpr = expr.expressions().first;
  auto rexpr = expr.expressions().second;
  return {make_const_variable<symbol>(std::move(lexpr)),
          make_const_variable<symbol>(std::move(rexpr))};
}

template <char symbol, typename Op, typename LHS>
constexpr auto make_const_variable(const MonoExpression<Op, LHS> &expr)
    -> MonoExpression<Op, replace_matching_variable_as_const_t<symbol, LHS>> {
  return {make_const_variable<symbol>(expr.expressions())};
}

template <typename T, char C, std::size_t N>
constexpr void make_labels_array(const Variable<T, C> &,
                                 std::array<char, N> &out, std::size_t &index) {
  out[index++] = C;
}

template <typename T, std::size_t N>
constexpr void make_labels_array(const Constant<T> &, std::array<char, N> &,
                                 std::size_t &) {
  // no-op
}

template <typename Op, typename LHS, typename RHS, std::size_t N>
constexpr void make_labels_array(const Expression<Op, LHS, RHS> &expr,
                                 std::array<char, N> &out, std::size_t &index) {
  make_labels_array(expr.expressions().first, out, index);
  make_labels_array(expr.expressions().second, out, index);
}

template <char symbol, typename Expr> struct constify_unmatched_var;

template <char symbol, typename T>
struct constify_unmatched_var<symbol, Variable<T, symbol>> {
  using type = Variable<T, symbol>;
};

template <char symbol, typename T, char othersymbol>
struct constify_unmatched_var<symbol, Variable<T, othersymbol>> {
  using type = Constant<T>;
};

template <char symbol, typename T>
struct constify_unmatched_var<symbol, Constant<T>> {
  using type = Constant<T>;
};

template <char symbol, typename Op, typename LHS, typename RHS>
struct constify_unmatched_var<symbol, Expression<Op, LHS, RHS>> {
  using type =
      Expression<Op, typename constify_unmatched_var<symbol, LHS>::type,
                 typename constify_unmatched_var<symbol, RHS>::type>;
};

template <char symbol, typename Expr>
using constify_unmatched_var_t =
    typename constify_unmatched_var<symbol, Expr>::type;

template <char symbol, typename T>
constexpr auto make_all_constant_except(const Variable<T, symbol> &v) {
  return v;
}

template <char symbol, typename T, char othersymbol>
constexpr auto make_all_constant_except(const Variable<T, othersymbol> &var)
    -> std::enable_if_t<(symbol != othersymbol), Constant<T>> {
  return Constant<T>{var};
}

template <char Symbol, typename T>
constexpr auto make_all_constant_except(const Constant<T> &c) {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_all_constant_except(const Expression<Op, LHS, RHS> &expr)
    -> constify_unmatched_var_t<symbol, Expression<Op, LHS, RHS>> {
  auto new_lhs = make_all_constant_except<symbol>(expr.expressions().first);
  auto new_rhs = make_all_constant_except<symbol>(expr.expressions().second);
  return {new_lhs, new_rhs};
}

template <char symbol, typename Op, typename Expr>
constexpr auto make_all_constant_except(const MonoExpression<Op, Expr> &expr)
    -> constify_unmatched_var_t<symbol, MonoExpression<Op, Expr>> {
  return make_all_constant_except<symbol>(expr.expressions());
}

template <typename T> struct extract_variable_symbols {
  using type = std::tuple<>;
};

template <typename T, char Symbol>
struct extract_variable_symbols<Variable<T, Symbol>> {
  using type = std::tuple<std::integral_constant<char, Symbol>>;
};

template <typename T>
struct extract_symbols_from_expr {
  using type = typename extract_variable_symbols<T>::type;
};

template <typename Op, typename LHS, typename RHS>
struct extract_symbols_from_expr<Expression<Op, LHS, RHS>> {
private:
  using left = typename extract_symbols_from_expr<LHS>::type;
  using right = typename extract_symbols_from_expr<RHS>::type;

public:
  using type =
      decltype(std::tuple_cat(std::declval<left>(), std::declval<right>()));
};

template <char... Cs>
struct charlist {};

template <typename Tuple>
struct tuple_to_charlist;

template <char... Cs>
struct tuple_to_charlist<std::tuple<std::integral_constant<char, Cs>...>> {
  using type = charlist<Cs...>;
};

template <typename Expr>
struct extract_charlist {
private:
  using as_tuple = typename extract_symbols_from_expr<Expr>::type;
public:
  using type = typename tuple_to_charlist<as_tuple>::type;
};
template <typename Tuple, typename Op, typename LHS, typename RHS, std::size_t... Is>
constexpr auto transform_tuple_chars_impl(const Tuple& chars, const Expression<Op, LHS, RHS>& expr, std::index_sequence<Is...>) {
  return std::make_tuple(
      make_all_constant_except<std::tuple_element_t<Is, Tuple>::value>(expr)...
  );
}

template <typename... Chars, typename Op, typename LHS, typename RHS>
constexpr auto transform_tuple_chars(const std::tuple<Chars...>& chars, const Expression<Op, LHS, RHS>& expr) {
  return transform_tuple_chars_impl(chars, expr, std::index_sequence_for<Chars...>{});
}
