/**
 * \file       lagrange.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <vector>

#include "polycalc/parameters.hpp"

namespace polycalc {
namespace interpolation {

template <typename T, typename P = DefaultParameters<T>>
class Lagrange {
   public:
    using value_type = T;
    using params     = P;
    using size_type  = std::size_t;

    Lagrange()                      = default;
    Lagrange(const Lagrange& other) = default;
    Lagrange(Lagrange&& other)      = default;
    ~Lagrange()                     = default;
    Lagrange& operator=(const Lagrange& other) = default;
    Lagrange& operator=(Lagrange&& other) = default;

    /**
     * Construct using the nodal locations of interpolation
     *
     * @param points Iterable container of point locations
     */
    template <typename IterableContainer>
    Lagrange(const IterableContainer& points);

    /**
     * Assign the nodal locations of interpolation
     *
     * @param points Iterable container of point locations
     */
    template <typename IterableContainer>
    void reset(const IterableContainer& points);

    /**
     * Return the i'th polynomial evaluated at location x
     *
     * @param i Polynomial to calculate
     * @param x Location to evaluate at
     * @return i'th polynomial evaluated at location x
     **/
    value_type eval(const size_type i, const value_type x) const;

    /**
     * Return the derivative for the i'th polynomial evaluated at location x
     *
     * @param i Polynomial to calculate
     * @param x Location to evaluate at
     * @return Derivative of i'th polynomial evaluated at location x
     **/
    value_type ddx(const size_type i, const value_type x) const;

    /**
     * Number of Nodes
     *
     * @return Number of nodes within Lagrange polynomial
     */
    size_type size() const { return x_.size(); }

    /**
     * Degree of interpolating polynomial
     *
     * @return Degree of Lagrange polynomial
     */
    size_type degree() const { return this->size() - 1; }

    /**
     * Location of the i'th node
     *
     * @param i Node to return
     * @return Location x of i'th node
     **/
    value_type point(const size_type i) const {
        assert(i < this->size());
        return x_[i];
    }

   protected:
    /// Precompute the denominators
    void precompute_();

   private:
    std::vector<value_type> x_;  ///< Node Locations
    std::vector<value_type> w_;  ///< Barycentric Weights
};

namespace detail {

/// Perform Subtraction with less Subtractive Cancellation
/**
 * When values are close it strips off the simular parts and scales up the
 * remainder to perform the subtraction.  Then unscales the difference of the
 * remainder to proces a "better" answer.
 *
 * @note
 * Expensive so only use when necessary.
 */
// template <typename T>
// T subtract(const T x, const T y) {

//     // If the values are "far" away then skip
//     if ( static_cast<T>(1.0E-4) < std::abs(x - y) ){
//         return (x - y);
//     }

//     using int_type  = long int;
//     using real_type = long double;

//     constexpr auto eps  = std::numeric_limits<real_type>::epsilon();
//     const auto digits   = static_cast<int_type>(std::abs(std::floor(std::log10(eps))) - 1);
//     const auto sub_expn = static_cast<int_type>(std::floor(std::log10(std::abs(x - y))));
//     assert(sub_expn < 0);

//     const real_type scaled_x  = x * std::pow(10ULL, -sub_expn);              // Scale up by non-overlaping digits
//     const real_type remain_x  = scaled_x - std::floor(scaled_x);             // Get remainder that is different
//     const real_type scaled_rx = remain_x * std::pow(10ULL, digits+sub_expn); // Scale up remainder to max precision
//     const real_type trimed_rx = std::round(scaled_rx);                       // Remove junk that cannot be
//     represented

//     const real_type scaled_y  = y * std::pow(10ULL, -sub_expn);              // Scale up by non-overlaping digits
//     const real_type remain_y  = scaled_y - std::floor(scaled_y);             // Get remainder that is different
//     const real_type scaled_ry = remain_y * std::pow(10ULL, digits+sub_expn); // Scale up remainder to may precision
//     const real_type trimed_ry = std::round(scaled_ry);                       // Remove junk that cannot be
//     represented

//     return static_cast<T>(trimed_rx - trimed_ry) * std::pow(10ULL, -digits); // Unscale the difference of remainder
// };

template <typename T>
T subtract(const T x, const T y) {
    return (x - y);
};

} /* namespace detail */

template <typename T, typename P>
template <typename IterableContainer>
Lagrange<T, P>::Lagrange(const IterableContainer& points) {
    this->reset(points);
}

template <typename T, typename P>
template <typename IterableContainer>
void Lagrange<T, P>::reset(const IterableContainer& points) {
    x_.clear();
    std::copy(points.begin(), points.end(), std::back_inserter(x_));
    this->precompute_();
}

template <typename T, typename P>
typename Lagrange<T, P>::value_type Lagrange<T, P>::eval(const size_type i, const value_type x) const {
    assert((0 <= i) and (i < x_.size()));
    const size_type sz = x_.size();

    value_type lx = 1;
    for (size_type n = 0; n < sz; ++n) {
        lx *= detail::subtract(x, x_[n]);
    }
    value_type dx = detail::subtract(x, x_[i]);
    return (std::abs(dx) < params::TOL) ? 1: lx*w_[i]/dx;
}

template <typename T, typename P>
typename Lagrange<T, P>::value_type Lagrange<T, P>::ddx(const size_type j, const value_type x) const {
    assert((0 <= j) and (j < x_.size()));

    constexpr value_type one = 1;

    value_type prod;
    value_type lj1     = 0;
    const size_type sz = x_.size();
    for (size_type i = 0; i < sz; ++i) {
        if (i != j) {
            prod = one / detail::subtract(x_[j], x_[i]);
            for (size_type m = 0; m < sz; ++m) {
                if (not(m == i) && not(m == j)) {
                    prod *= detail::subtract(x, x_[m]) / detail::subtract(x_[j], x_[m]);
                }
            }
            lj1 += prod;
        }
    }
    return lj1;
}

template <typename T, typename P>
void Lagrange<T, P>::precompute_() {
    const size_type sz = x_.size();

    w_.resize(sz);
    std::fill(w_.begin(), w_.end(), 1);
    for (size_type j = 1; j < sz; ++j) {
        for (size_type k = 0; k < j; ++k) {
            auto dx = detail::subtract(x_[k], x_[j]);
            assert(std::abs(dx) > params::TOL);  // Repeat Points
            w_[k] *= +dx;
            w_[j] *= -dx;
        }
    }
    for (size_type j = 0; j < sz; ++j) {
        w_[j] = static_cast<value_type>(1.0) / w_[j];
    }
}

}  // namespace interpolation
}  // namespace polycalc
