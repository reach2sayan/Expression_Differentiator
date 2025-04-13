//
// Created by sayan on 4/13/25.
//

#include "matrix.hpp"
#include "operations.hpp"
#include "values.hpp"

#include <gtest/gtest.h>

std::array<int, 16> data1 = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
std::array<int, 16> data2 = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
auto manual_add(auto a, auto b) {
  std::array<int, 16> result;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      result[i * 16 + j] = a[i * 16 + j] + b[i * 16 + j];
    }
  }
  return result;
}

TEST(ExpressionTest, SumTest) {
  double a = 1, b = 2, c = 3;
  auto sum_exp = Sum<int>(a, Sum<int>(b, c));
  EXPECT_EQ(sum_exp, 6);
}

TEST(ExpressionTest, ExpTest) {
  auto exp_exp = Exp<int>(2, 4);
  EXPECT_EQ(exp_exp, 16);
}

TEST(ExpressionTest, ExpSum) {
  auto target = Exp<int>(Sum<int>(1, 2), 2);
  EXPECT_EQ(target, 9);
}

TEST(ExpressionTest, Combination) {
  auto target = Sum<int>(Exp<int>(2, Sum<int>(1, Sum<int>(2, 3))), 1);
  EXPECT_EQ(target, 65);
}

TEST(ExpressionTest, ConstantTest) {
  auto target = Constant<int>(1);
  auto derv = target.derivative();
  EXPECT_EQ(derv,0);
  EXPECT_EQ(target,1);
}

TEST(ExpressionTest, DerivativeTest) {
  auto target = Constant<int>(1);
  auto derv = target.derivative();
  EXPECT_EQ(derv,0);
  EXPECT_EQ(target,1);
}