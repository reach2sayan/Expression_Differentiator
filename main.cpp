#include <iostream>

#include "operations.hpp"
#include "matrix.hpp"

std::array<int,16> data1 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
std::array<int,16> data2 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

int main() {
	matrix<int,4,4> m1(data1);
	matrix<int,4,4> m2(data2);
	auto target = Sum<matrix<int,4,4>>(m1,m2);
	auto multarget = Multiply<matrix<int,4,4>>(m1,m2);
	//auto neg_mult = Negate<matrix<int,4,4>>(multarget);
	std::cout << multarget.eval() << std::endl;
	//std::cout << neg_mult << std::endl;
}