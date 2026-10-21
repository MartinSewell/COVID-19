//
// DTW.hpp
//
// Copyright (c) 2019 Charles Jekel
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//


#ifndef MARTIN_REVIEWED_DTW_HPP_INCLUDED
#define MARTIN_REVIEWED_DTW_HPP_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Reviewed 21 October 2026. Public names and the accumulated-cost DTW rule
// are retained. This header is self-contained and safe in multiple .cpp files.
inline const std::string DTW_VERSION = "0.0.1";

namespace DTW {
using Matrix = std::vector<std::vector<double>>;

namespace detail {
inline void check_p(double p) {
    if (!std::isfinite(p) || p <= 0.0)
        throw std::invalid_argument("DTW requires a finite, positive p.");
}
inline std::size_t check_matrix(const Matrix& values) {
    if (values.empty() || values.front().empty())
        throw std::invalid_argument("DTW requires non-empty sequences and dimensions.");
    const std::size_t dimensions = values.front().size();
    for (const auto& row : values) {
        if (row.size() != dimensions)
            throw std::invalid_argument("DTW rows must have a consistent number of dimensions.");
        for (double value : row)
            if (!std::isfinite(value))
                throw std::invalid_argument("DTW inputs must contain only finite values.");
    }
    return dimensions;
}
inline void check_sequences(const Matrix& a, const Matrix& b, double p) {
    check_p(p);
    if (check_matrix(a) != check_matrix(b))
        throw std::invalid_argument("a and b must have the same number of dimensions.");
}
// Called only after validation; avoid rechecking/copying coordinates at each cell.
inline double point_distance(const std::vector<double>& a,
                             const std::vector<double>& b, double p) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum += std::pow(std::abs(a[i] - b[i]), p);
    const double result = std::pow(sum, 1.0 / p);
    if (!std::isfinite(result))
        throw std::overflow_error("DTW point distance exceeds the finite numeric range.");
    return result;
}
inline double finite_distance(double distance) {
    if (!std::isfinite(distance))
        throw std::overflow_error("DTW accumulated distance exceeds the finite numeric range.");
    return distance;
}
} // namespace detail

inline double p_norm(const std::vector<double>& a,
                     const std::vector<double>& b, double p) {
    detail::check_p(p);
    if (a.size() != b.size())
        throw std::invalid_argument("p_norm requires equal-length vectors.");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::invalid_argument("p_norm inputs must contain only finite values.");
    return detail::point_distance(a, b, p);
}

// Exact unrestricted DTW with the original unnormalised accumulated cost.
// Two rows suffice when only the distance is requested; no full n-by-m matrix.
inline double dtw_distance_only(const Matrix& a, const Matrix& b, double p) {
    detail::check_sequences(a, b, p);
    std::vector<double> previous(b.size(), 0.0), current(b.size(), 0.0);
    previous[0] = detail::point_distance(a[0], b[0], p);
    for (std::size_t j = 1; j < b.size(); ++j)
        previous[j] = previous[j-1] + detail::point_distance(a[0], b[j], p);
    for (std::size_t i = 1; i < a.size(); ++i) {
        current[0] = previous[0] + detail::point_distance(a[i], b[0], p);
        for (std::size_t j = 1; j < b.size(); ++j)
            current[j] = detail::point_distance(a[i], b[j], p)
                + std::min(std::min(previous[j], current[j-1]), previous[j-1]);
        previous.swap(current);
    }
    return detail::finite_distance(previous.back());
}

// Assemble accumulated costs, including the origin cell c[0][0].
inline Matrix dtw_vector_assemble(Matrix d, const Matrix& c) {
    const std::size_t columns = detail::check_matrix(c);
    if (d.size() != c.size())
        throw std::invalid_argument("DTW cost and accumulated matrices must have the same shape.");
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (d[i].size() != columns)
            throw std::invalid_argument("DTW cost and accumulated matrices must have the same shape.");
        for (double cost : c[i])
            if (cost < 0.0)
                throw std::invalid_argument("DTW point costs must be non-negative.");
    }
    d[0][0] = c[0][0];
    for (std::size_t i = 1; i < d.size(); ++i)
        d[i][0] = d[i-1][0] + c[i][0];
    for (std::size_t j = 1; j < columns; ++j)
        d[0][j] = d[0][j-1] + c[0][j];
    for (std::size_t i = 1; i < d.size(); ++i)
        for (std::size_t j = 1; j < columns; ++j)
            d[i][j] = c[i][j] + std::min(std::min(d[i-1][j], d[i][j-1]), d[i-1][j-1]);
    detail::finite_distance(d.back().back());
    return d;
}

class DTW {
public:
    Matrix a_vector, b_vector;
    int a_data = 0, n_dim = 0, b_data = 0;
    double p = 2.0, distance = 0.0;
    Matrix dtw_vector, pairwise_vector;

    DTW(const Matrix& a, const Matrix& b, double norm_p)
        : a_vector(a), b_vector(b), p(norm_p) {
        detail::check_sequences(a_vector, b_vector, p);
        const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (a_vector.size() > maximum || b_vector.size() > maximum
                || a_vector.front().size() > maximum)
            throw std::length_error("DTW sequence dimensions exceed the integer path index range.");
        a_data = static_cast<int>(a_vector.size());
        b_data = static_cast<int>(b_vector.size());
        n_dim = static_cast<int>(a_vector.front().size());
        pairwise_vector.assign(a_vector.size(), std::vector<double>(b_vector.size(), 0.0));
        for (std::size_t i = 0; i < a_vector.size(); ++i)
            for (std::size_t j = 0; j < b_vector.size(); ++j)
                pairwise_vector[i][j] = detail::point_distance(a_vector[i], b_vector[j], p);
        dtw_vector = dtw_vector_assemble(
            Matrix(a_vector.size(), std::vector<double>(b_vector.size(), 0.0)), pairwise_vector);
        distance = dtw_vector.back().back();
    }

    std::vector<std::vector<int>> path() const {
        // Public compatibility fields can be modified by callers; guard indexing.
        if (a_data <= 0 || b_data <= 0 || dtw_vector.size() != static_cast<std::size_t>(a_data))
            throw std::invalid_argument("Invalid DTW path matrix.");
        for (const auto& row : dtw_vector)
            if (row.size() != static_cast<std::size_t>(b_data))
                throw std::invalid_argument("Invalid DTW path matrix.");
        int i = a_data - 1, j = b_data - 1;
        std::vector<std::vector<int>> result = {{i, j}};
        while (i > 0 || j > 0) {
            if (i == 0) --j;
            else if (j == 0) --i;
            else {
                const double step = std::min(std::min(dtw_vector[i-1][j], dtw_vector[i][j-1]),
                                             dtw_vector[i-1][j-1]);
                // Preserve the original vertical, horizontal, diagonal tie order.
                if (step == dtw_vector[i-1][j]) --i;
                else if (step == dtw_vector[i][j-1]) --j;
                else { --i; --j; }
            }
            result.push_back({i, j});
        }
        std::reverse(result.begin(), result.end());
        return result;
    }
};
} // namespace DTW
#endif
