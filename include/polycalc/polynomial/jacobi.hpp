/**
 * \file       jacobi.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <vector>

#include "polycalc/parameters.hpp"

namespace polycalc {
namespace polynomial {

template <typename T, typename P = DefaultParameters<T>>
struct Jacobi {
    using value_type = T;
    using params     = P;
    using size_type  = std::size_t;

    Jacobi()                    = delete;
    Jacobi(const Jacobi& other) = default;
    Jacobi(Jacobi&& other)      = default;
    ~Jacobi()                   = default;

    Jacobi(const value_type alpha, const value_type beta) : alpha_(alpha), beta_(beta) {}

    Jacobi& operator=(const Jacobi& other) = default;
    Jacobi& operator=(Jacobi&& other) = default;

    /**
     * Evaluate n'th order Jacobi polynomial at location x.
     *
     * @param n Order of Jacobi polnomial
     * @param x Location of value
     */
    value_type eval(const size_type n, const value_type x) const {
        assert((x <= 1) && (x >= -1));
        return this->fast_eval_(n, x);
    }

    /**
     * Evaluate k'th derivative of n'th order Jacobi polynomial at location x.
     *
     * @param k Order of derivative for Jacobi polnomial
     * @param n Order of Jacobi polnomial
     * @param x Location of derivative
     */
    value_type derivative(const size_type k, const size_type n, const value_type x) const {
        assert((x <= 1) && (x >= -1));
        return this->fast_derivative_(k, n, x);
    }

    /**
     * Evaluate derivative of n'th order Jacobi polynomial at location x.
     *
     * @param n Order of Jacobi polnomial
     * @param x Location of derivative
     */
    value_type ddx(const size_type n, const value_type x) const {
        assert((x <= 1) && (x >= -1));
        return this->fast_derivative_(1, n, x);
    }

    /**
     * Calculate the n zeros of the Jacobi polynomial
     *
     * @param n Number of zeros to return
     *
     */
    std::vector<value_type> zeros(const size_type n) const {
        assert(n > 0);
        std::vector<T> x(n, 0);
        this->deflation_zeros_(n, x);
        return x;
    }

   private:
    value_type fast_eval_(const size_type n, const value_type x) const;
    value_type fast_derivative_(const size_type k, const size_type n, const value_type x) const;
    void deflation_zeros_(const size_type n, std::vector<value_type>& x) const;

    value_type alpha_;  ///< Alpha Paramater
    value_type beta_;   ///< Beta Paramater
};




template <typename T, typename P>
typename Jacobi<T, P>::value_type Jacobi<T, P>::fast_eval_(const size_type n, const value_type x) const {
    if (0 == n) {
        return static_cast<value_type>(1);
    }

    constexpr value_type one  = 1;
    constexpr value_type two  = 2;
    const value_type apb  = alpha_ + beta_;

    value_type y0 = one;
    value_type y1 = (alpha_+one) + (apb+two)*(x-one)/two;

    value_type yk = y1;
    value_type k = 2;
    value_type k_max = n*(1+std::numeric_limits<value_type>::epsilon());
    while(k < k_max){
        value_type den = two*k*(k+apb)*(two*k+apb-two);
        value_type gm1 = (two*k+apb-one)*( (two*k+apb)*(two*k+apb-two)*x + alpha_*alpha_ -beta_*beta_);
        value_type gm0 = -two*(k+alpha_-one)*(k+beta_-one)*(two*k+apb);
        yk = (gm1*y1 + gm0*y0)/den;
        y0 = y1;
        y1 = yk;
        k += 1;
    }
    return yk;



}

template <typename T, typename P>
typename Jacobi<T, P>::value_type Jacobi<T, P>::fast_derivative_(const size_type k, const size_type n, const value_type x) const {

    if ((0 == n) or (k > n)) {
        return static_cast<value_type>(0);
    }

    value_type scale = 1;
    for(auto i = 1; i <= k; ++i) {
        scale *= (alpha_ + beta_ + n + i)/2;
    }
    Jacobi jac(alpha_ + k, beta_ + k);
    return scale * jac.eval(n - k, x);
}


template <typename T, typename P>
void Jacobi<T, P>::deflation_zeros_(const size_type n, std::vector<value_type>& x) const {
    assert(n > 0);
    assert(x.size() >= n);

    const size_type STOP = params::MAX_ITER;
    const value_type TOL = params::TOL;
    const value_type den = M_PI / (static_cast<value_type>(2 * n));

    // Loop over for each Zero
    for (size_type k = 0; k < n; ++k) {
        // Use explicit formula for the zeros of the
        // Chebyshev polynomial (a=b=-1/2) as the initial guess for
        // the root of all Jacobi polynomials
        value_type zero = -std::cos((2 * k + 1) * den);

        // Perform Newton Iterations
        value_type poly, pder;
        value_type pinc = 10 * TOL;
        for (size_type j = 0; (j < STOP) and (std::abs(pinc) > TOL); ++j) {
            poly = this->eval(n,zero);
            pder = this->derivative(1, n, zero);

            // Polynomial Deflation
            // Factor out the previous zeros
            value_type sum = 0;
            for (size_type i = 0; i < k; ++i) {
                sum += 1 / (zero - x[i]);
            }
            pinc = poly / (pder - sum * poly);
            zero -= pinc;
        }
        x[k] = zero;
    }
}

} /* namespace polynomial */
} /* namespace polycalc */
