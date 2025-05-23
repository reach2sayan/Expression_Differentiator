//
// Created by sayan on 4/18/25.
//

#pragma once
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

template <char symbol, typename Op, typename Expr>
struct constify_unmatched_var<symbol, MonoExpression<Op, Expr>> {
  using type =
      MonoExpression<Op, typename constify_unmatched_var<symbol, Expr>::type>;
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

template <typename T, typename sorted> struct insert_sorted;

template <typename T> struct insert_sorted<T, std::tuple<>> {
  using type = std::tuple<T>;
};

template <char A, char B>
constexpr bool operator==(std::integral_constant<char, A>,
                          std::integral_constant<char, B>) {
  return A == B;
}

template <char A, char B>
constexpr bool operator<(std::integral_constant<char, A>,
                         std::integral_constant<char, B>) {
  return A < B;
}
template <typename T, typename head, typename... tail>
struct insert_sorted<T, std::tuple<head, tail...>> {
  using type = std::conditional_t<
      (T{} < head{}), std::tuple<T, head, tail...>,
      decltype(std::tuple_cat(
          std::tuple<head>{},
          typename insert_sorted<T, std::tuple<tail...>>::type{}))>;
};

template <typename> struct sort_tuple;

template <> struct sort_tuple<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename head, typename... tail>
struct sort_tuple<std::tuple<head, tail...>> {
  using sorted_tail = typename sort_tuple<std::tuple<tail...>>::type;
  using type = typename insert_sorted<head, sorted_tail>::type;
};

template <typename Tuple> using sort_tuple_t = typename sort_tuple<Tuple>::type;

template <typename Tuple> struct unique_tuple;

template <> struct unique_tuple<std::tuple<>> {
  using type = std::tuple<>;
};

template <typename T> struct unique_tuple<std::tuple<T>> {
  using type = std::tuple<T>;
};

template <typename A, typename B, typename... Rest>
struct unique_tuple<std::tuple<A, B, Rest...>> {
  using tail = std::conditional_t<A{} == B{}, std::tuple<B, Rest...>,
                                  std::tuple<B, Rest...>>;

  using rest_unique = typename unique_tuple<tail>::type;

  using type = std::conditional_t<A{} == B{}, rest_unique,
                                  decltype(std::tuple_cat(std::tuple<A>{},
                                                          rest_unique{}))>;
};

template <typename T> using unique_tuple_t = typename unique_tuple<T>::type;

template <typename T, typename Tuple> struct tuple_contains;

template <typename T>
struct tuple_contains<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Rest>
struct tuple_contains<T, std::tuple<U, Rest...>>
    : tuple_contains<T, std::tuple<Rest...>> {};

template <typename T, typename... Rest>
struct tuple_contains<T, std::tuple<T, Rest...>> : std::true_type {};

template <typename T, typename Tuple> struct tuple_append_unique {
  using type = std::conditional_t<tuple_contains<T, Tuple>::value, Tuple,
                                  decltype(std::tuple_cat(
                                      std::declval<Tuple>(),
                                      std::declval<std::tuple<T>>()))>;
};

template <typename first, typename second> struct tuple_union;

template <typename first> struct tuple_union<first, std::tuple<>> {
  using type = first;
};

template <typename first, typename head, typename... tail>
struct tuple_union<first, std::tuple<head, tail...>> {
private:
  using with_head = typename tuple_append_unique<head, first>::type;

public:
  using type = typename tuple_union<with_head, std::tuple<tail...>>::type;
};

template <typename... tuples> struct tuple_union_variadic;

template <> struct tuple_union_variadic<> {
  using type = std::tuple<>;
};

template <typename first> struct tuple_union_variadic<first> {
  using type = first;
};

template <typename first, typename second, typename... tail>
struct tuple_union_variadic<first, second, tail...> {
  using type =
      typename tuple_union_variadic<typename tuple_union<first, second>::type,
                                    tail...>::type;
};

template <typename... tuples>
using tuple_union_t = typename tuple_union_variadic<tuples...>::type;

template <typename first, typename second> struct tuple_difference;

template <typename second> struct tuple_difference<std::tuple<>, second> {
  using type = std::tuple<>;
};

template <typename head, typename... tail, typename second>
struct tuple_difference<std::tuple<head, tail...>, second> {
private:
  using result_tail =
      typename tuple_difference<std::tuple<tail...>, second>::type;

public:
  using type = std::conditional_t<tuple_contains<head, second>::value,
                                  result_tail, // Skip it
                                  decltype(std::tuple_cat(
                                      std::declval<std::tuple<head>>(),
                                      std::declval<result_tail>()))>;
};

template <typename first, typename second>
using tuple_difference_t = typename tuple_difference<first, second>::type;

template <typename T> struct extract_variable_symbols {
  using type = std::tuple<>;
};

template <typename T, char symbol>
struct extract_variable_symbols<Variable<T, symbol>> {
  using type = std::tuple<std::integral_constant<char, symbol>>;
};

template <typename T> struct extract_symbols_from_expr {
  using type = typename extract_variable_symbols<T>::type;
};

template <typename Op, typename LHS, typename RHS>
struct extract_symbols_from_expr<Expression<Op, LHS, RHS>> {
private:
  using left = typename extract_symbols_from_expr<LHS>::type;
  using right = typename extract_symbols_from_expr<RHS>::type;

public:
  using type = unique_tuple_t<sort_tuple_t<decltype(std::tuple_cat(
      std::declval<left>(), std::declval<right>()))>>;
};

template <typename Op, typename Expr>
struct extract_symbols_from_expr<MonoExpression<Op, Expr>> {
private:
  using left = typename extract_symbols_from_expr<Expr>::type;

public:
  using type = unique_tuple_t<
      sort_tuple_t<decltype(std::tuple_cat(std::declval<left>()))>>;
};

template <size_t value> struct idx_t : std::integral_constant<size_t, value> {};
#define IDX(value)                                                             \
  idx_t<value> {}

template <typename head, typename... tail>
constexpr static inline bool all_tuple_type_same =
    (std::is_same_v<typename head::value_type, typename tail::value_type> &&
     ...);
