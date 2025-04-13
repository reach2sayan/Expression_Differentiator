//
// Created by sayan on 4/13/25.
//

#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP
//
// Created by sayan on 4/13/25.
//
#include "matrix.hpp"
#include <cmath>
#include <type_traits>

#include "expressions.hpp"

enum class OpType : short {
  Unary = 0,
  Binary = 1,
};

template <typename T> using unary_op_func = T(const T &);
template <typename T> using binary_op_func = T(const T &, const T &);

template <typename T, OpType type> struct Op {
  using value_type = T;
  constexpr static OpType op_type = type;
};

template <typename T, unary_op_func<T> func, char symbol>
struct UnaryOp : Op<T, OpType::Unary> {
  using value_type = Op<T, OpType::Unary>::value_type;
  template <typename Expression1>
  static void print(std::ostream &out, const Expression1 &e1) {
    out << symbol << e1;
  }

  template <typename Expression1>
  constexpr static auto eval(const Expression1 &e1) {
    return std::invoke(func, e1);
  }
};

template <typename T, binary_op_func<T> func, char symbol>
struct BinaryOp : Op<T, OpType::Binary> {
  using value_type = Op<T, OpType::Binary>::value_type;
  template <typename Expression1, typename Expression2>
  static void print(std::ostream &out, const Expression2 &e1,
                    const Expression1 &e2) {
    out << e1 << symbol << e2;
  }
  template <typename Expression1, typename Expression2>
  constexpr static auto eval(const Expression1 &e1, const Expression2 &e2) {
    return std::invoke(func, e1, e2);
  }
};

template <typename Op> struct Derivative {
  constexpr static size_t count = [](OpType op) {
    return std::to_underlying(op) + 1;
  }(Op::op_type);
};

template <typename T>
struct ExpOp
    : BinaryOp<T, [](const T &a, const T &b) -> T { return std::pow(a, b); },
               '^'> {
  template <typename Expression1, typename Expression2>
  constexpr static auto derivative(const Expression1 &e1,
                                   const Expression2 &e2);
};

static_assert(ExpOp<int>::op_type == OpType::Binary);
static_assert(Derivative<ExpOp<int>>::count == 2);

template <typename T>
struct SumOp
    : BinaryOp<T, [](const T &a, const T &b) -> T { return a + b; }, '+'> {
  template <typename Expression1, typename Expression2>
  constexpr static auto derivative(const Expression1 &e1,
                                   const Expression2 &e2);
};

template <typename T, typename Expression1, typename Expression2>
constexpr inline auto Sum(Expression1 e1, Expression2 e2) {
  return Expression<SumOp<T>, Expression1, Expression2>(std::move(e1),
                                                        std::move(e2));
}

template <typename T>
template <typename Expression1, typename Expression2>
constexpr auto SumOp<T>::derivative(const Expression1 &e1,
                                    const Expression2 &e2) {
  using e1_deriv_t = decltype(e1.derivative());
  using e2_deriv_t = decltype(e2.derivative());
  return Sum<T, e1_deriv_t, e2_deriv_t>(e1.derivative(), e2.derivative());
}

template <typename T>
struct MultiplyOp
    : BinaryOp<T, +[](const T &a, const T &b) -> T { return a * b; }, '*'> {};

template <typename T>
struct NegateOp : UnaryOp<T, +[](const T &a) -> T { return -a; }, '-'> {};

template <typename T, typename Expression1, typename Expression2>
constexpr inline auto Multiply(Expression1 e1, Expression2 e2) {
  return Expression<MultiplyOp<T>, Expression1, Expression2>(std::move(e1),
                                                             std::move(e2));
}

template <typename T, typename Expression1>
constexpr inline auto Negate(Expression1 e) {
  return Expression<NegateOp<T>, Expression1>(std::move(e));
}

template <typename T, typename Expression1, typename Expression2>
constexpr inline auto Exp(Expression1 e1, Expression2 e2) {
  return Expression<ExpOp<T>, Expression1, Expression2>(std::move(e1),
                                                        std::move(e2));
}

#endif // OPERATIONS_HPP
