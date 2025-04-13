//
// Created by sayan on 4/13/25.
//

#ifndef GENERATORS_H
#define GENERATORS_H

#include <generator>
#include <memory>
#include "matrix.hpp"

enum class SpanTypes { ROW, COLUMN };

template<typename T, size_t N, size_t M>
struct BaseSpanGenerator {
	size_t index;
	std::shared_ptr<matrix<T, N, M>> mat;

	explicit BaseSpanGenerator(const matrix<T, N, M> &b, size_t idx) :
		index{idx}, mat{std::make_shared<matrix<T, N, M>>(b)} {}
	[[nodiscard]] virtual std::generator<T &> next() const = 0;
	virtual ~BaseSpanGenerator() = default;
};

template<typename T, size_t N, size_t M, SpanTypes type>
struct SpanGenerator : BaseSpanGenerator<T, N, M> {};

template<typename T, size_t N, size_t M>
struct SpanGenerator<T, N, M, SpanTypes::ROW> : BaseSpanGenerator<T, N, M> {
	using BaseSpanGenerator<T, N, M>::mat;
	using BaseSpanGenerator<T, N, M>::index;
	explicit SpanGenerator(const matrix<T, N, M> &b, size_t idx) : BaseSpanGenerator<T, N, M>(b, idx) {}
	[[nodiscard]] std::generator<T &> next() const override;
};

template<typename T, size_t N, size_t M>
struct SpanGenerator<T, N, M, SpanTypes::COLUMN> : BaseSpanGenerator<T, N, M> {
	using BaseSpanGenerator<T, N, M>::mat;
	using BaseSpanGenerator<T, N, M>::index;
	constexpr explicit SpanGenerator(const matrix<T, N, M> &b, size_t idx) : BaseSpanGenerator<T, N, M>(b, idx) {}
	[[nodiscard]] std::generator<T &> next() const override;
};

template<typename T, size_t N, size_t M>
std::generator<T &> SpanGenerator<T, N, M, SpanTypes::ROW>::next() const {
	const auto col = std::submdspan(mat->mat, index, std::full_extent);
	for (size_t i = 0; i < col.extent(0); ++i) {
		co_yield col[i];
	}
}

template<typename T, size_t N, size_t M>
std::generator<T &> SpanGenerator<T, N, M, SpanTypes::COLUMN>::next() const {
	const auto col = std::submdspan(mat->mat, std::full_extent, index);
	for (size_t i = 0; i < col.extent(0); ++i) {
		co_yield col[i];
	}
}
#endif // GENERATORS_H
