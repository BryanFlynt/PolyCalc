/**
 * \file       gauss_jacobi.hpp
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
class GaussJacobi {
   public:
    using value_type = T;
    using params     = P;
    using size_type  = std::size_t;
    using polynomial = ::polycalc::polynomial::Jacobi<T, P>;

    GaussJacobi()                         = delete;
    GaussJacobi(const GaussJacobi& other) = default;
    GaussJacobi(GaussJacobi&& other)      = default;
    ~GaussJacobi()                        = default;
    GaussJacobi& operator=(const GaussJacobi& other) = default;
    GaussJacobi& operator=(GaussJacobi&& other) = default;

    GaussJacobi(const value_type a, const value_type b) : poly_(a, b) {}

    /** Quadrature Locations
     *
     * Returns the Gauss-Jacobi quadrature locations at n locations.
     */
    std::vector<value_type> zeros(const unsigned n) const;

    /** Quadrature Weights
     *
     * Returns the Gauss-Jacobi weights at n locations.
     */
    std::vector<value_type> weights(const unsigned n) const;

   private:
    polynomial poly_;
};

template <typename T, typename P>
std::vector<typename GaussJacobi<T, P>::value_type> GaussJacobi<T, P>::zeros(const unsigned n) const {
    assert(n > 0);
    return poly_.zeros(n);
}

template <typename T, typename P>
std::vector<typename GaussJacobi<T, P>::value_type> GaussJacobi<T, P>::weights(const unsigned n) const {
    assert(n > 0);

    // Get location of zeros
    auto z = this->zeros(n);

    // Evaluate Jacobi polynomial derivative at each zero
    std::vector<value_type> w(n);
    for (size_type i = 0; i < n; ++i) {
        w[i] = poly_.ddx(n, z[i]);
    }

    const value_type one = 1;
    const value_type two = 2;
    const value_type apb = poly_.alpha + poly_.beta;
    const value_type np1 = 1 + n;

    value_type fac;
    fac = std::pow(two, apb + one) * std::tgamma(poly_.alpha + np1) * std::tgamma(poly_.beta + np1);
    fac /= std::tgamma(np1) * std::tgamma(apb + np1);

    for (size_type i = 0; i < n; ++i) {
        w[i] = fac / (w[i] * w[i] * (one - z[i] * z[i]));
    }
    return w;
}

}  // namespace quadrature
}  // namespace polycalc
