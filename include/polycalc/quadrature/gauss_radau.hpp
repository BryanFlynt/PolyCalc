/**
 * \file       gauss_radau.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <cassert>
#include <vector>

#include "polycalc/parameters.hpp"
#include "polycalc/polynomial/jacobi.hpp"

namespace polycalc {
namespace quadrature {

template <typename T, typename P = DefaultParameters<T>>
class GaussRadau {
   public:
    using value_type = T;
    using params     = P;
    using size_type  = std::size_t;
    using polynomial = ::polycalc::polynomial::Jacobi<T, P>;

    GaussRadau()                        = delete;
    GaussRadau(const GaussRadau& other) = default;
    GaussRadau(GaussRadau&& other)      = default;
    ~GaussRadau()                       = default;
    GaussRadau& operator=(const GaussRadau& other) = default;
    GaussRadau& operator=(GaussRadau&& other) = default;

    GaussRadau(const value_type a, const value_type b) : alpha_(a), beta_(b) {}

    /** Quadrature Locations
     *
     * Returns the Gauss-Radau quadrature locations at n locations.
     */
    template <int Z1>
    std::vector<value_type> zeros(const unsigned n) const;

    /** Quadrature Weights
     *
     * Returns the Gauss-Jacobi weights at n Radau locations.
     */
    template <int Z1>
    std::vector<value_type> weights(const unsigned n) const;

   private:
    value_type alpha_;
    value_type beta_;
};

template <typename T, typename P>
template <int Z1>
std::vector<typename GaussRadau<T, P>::value_type> GaussRadau<T, P>::zeros(const unsigned n) const {
    assert(n > 0);

    if (n == 1) {
        return std::vector<value_type>(1, 0);
    }

    std::vector<value_type> x(n);

    if constexpr (Z1 == +1) {  // Z = +1
        polynomial jac(alpha_ + 1, beta_);
        auto zeros = jac.zeros(n - 1);
        std::copy(zeros.begin(), zeros.end(), x.begin());
        x[n - 1] = 1;
    } else if constexpr (Z1 == -1) {  // Z = -1
        polynomial jac(alpha_, beta_ + 1);
        auto zeros = jac.zeros(n - 1);
        std::copy(zeros.begin(), zeros.end(), x.begin() + 1);
        x[0] = -1;
    } else {
        assert(false);
    }
    return x;
}

template <typename T, typename P>
template <int Z1>
std::vector<typename GaussRadau<T, P>::value_type> GaussRadau<T, P>::weights(const unsigned n) const {
    assert(n > 0);

    if (n == 1) {
        return std::vector<value_type>(1, 2);
    }

    // Get location of zeros
    auto z = this->zeros<Z1>(n);

    // Evaluate Jacobi polynomial at each zero
    polynomial jac(alpha_, beta_);
    std::vector<value_type> w(n);
    for (size_type i = 0; i < n; ++i) {
        w[i] = jac.eval(n - 1, z[i]);
    }

    const value_type one = 1;
    const value_type two = 2;
    const value_type apb = alpha_ + beta_;
    const value_type np1 = 1 + n;
    value_type fac;

    if constexpr (Z1 == +1) {  // Z = +1
        fac = std::pow(two, apb) * std::tgamma(alpha_ + n) * std::tgamma(beta_ + n);
        fac /= std::tgamma(n) * (alpha_ + n) * std::tgamma(apb + n + 1);

        for (size_type i = 0; i < n; ++i) {
            w[i] = fac * (one + z[i]) / (w[i] * w[i]);
        }
        w[n - 1] *= (alpha_ + one);
    } else if constexpr (Z1 == -1) {  // Z = -1
        fac = std::pow(two, apb) * std::tgamma(alpha_ + n) * std::tgamma(beta_ + n);
        fac /= std::tgamma(n) * (beta_ + n) * std::tgamma(apb + n + 1);

        for (size_type i = 0; i < n; ++i) {
            w[i] = fac * (one - z[i]) / (w[i] * w[i]);
        }
        w[0] *= (beta_ + one);
    } else {
        assert(false);
    }
    return w;
}

}  // namespace quadrature
}  // namespace polycalc
