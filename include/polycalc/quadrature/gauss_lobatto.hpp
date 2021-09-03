/**
 * \file       gauss_lobatto.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <cassert>
#include <vector>

#include "parameters.hpp"
#include "polynomial/jacobi.hpp"

namespace polycalc {
namespace quadrature {

template <typename T, typename P = DefaultParameters<T>>
class GaussLobatto {
   public:
    using value_type = T;
    using params     = P;
    using size_type  = std::size_t;
    using polynomial = ::polycalc::polynomial::Jacobi<T, P>;

    GaussLobatto()                          = delete;
    GaussLobatto(const GaussLobatto& other) = default;
    GaussLobatto(GaussLobatto&& other)      = default;
    ~GaussLobatto()                         = default;
    GaussLobatto& operator=(const GaussLobatto& other) = default;
    GaussLobatto& operator=(GaussLobatto&& other) = default;

    GaussLobatto(const value_type a, const value_type b) : alpha_(a), beta_(b) {}

    /** Quadrature Locations
     *
     * Returns the Gauss-Lobatto quadrature locations at n locations.
     */
    std::vector<value_type> zeros(const unsigned n) const;

    /** Quadrature Weights
     *
     * Returns the Gauss-Jacobi weights at n Lobatto zeros.
     */
    std::vector<value_type> weights(const unsigned n) const;

   private:
    value_type alpha_;
    value_type beta_;
};

template <typename T, typename P>
std::vector<typename GaussLobatto<T, P>::value_type> GaussLobatto<T, P>::zeros(const unsigned n) const {
    assert(n > 0);

    // Zeros to return
    std::vector<value_type> x(n);

    if (1 == n) {
        x[0] = 0.0;
    } else if (2 == n) {
        x[0] = -1.0;
        x[1] = +1.0;
    } else {
        polynomial jac(alpha_ + 1, beta_ + 1);
        auto zeros = jac.zeros(n - 2);

        x[0] = -1.0;
        std::copy(zeros.begin(), zeros.end(), x.begin() + 1);
        x[n - 1] = +1.0;
    }
    return x;
}

template <typename T, typename P>
std::vector<typename GaussLobatto<T, P>::value_type> GaussLobatto<T, P>::weights(const unsigned n) const {
    assert(n > 0);
    using std::pow;
    using std::tgamma;

    // Weights to return
    std::vector<value_type> w(n);

    if (1 == n) {
        w[0] = 2.0;
    } else if (2 == n) {
        w[0] = 1.0;
        w[1] = 1.0;
    } else {
        // Get location of zeros
        auto z = this->zeros(n);

        // Evaluate Jacobi n-1 polynomial at each zero
        polynomial jac(alpha_, beta_);
        std::vector<value_type> w(n);
        for (size_type i = 0; i < n; ++i) {
            w[i] = jac.eval(n - 1, z[i]);
        }

        const value_type one = 1;
        const value_type two = 2;
        const value_type apb = alpha_ + beta_;

        value_type fac;
        fac = pow(two, apb + one) * tgamma(alpha_ + n) * tgamma(beta_ + n);
        fac /= (n - 1) * tgamma(n) * tgamma(alpha_ + beta_ + n + one);

        for (size_type i = 0; i < n; ++i) {
            w[i] = fac / (w[i] * w[i]);
        }
        w[0] *= (beta_ + one);
        w[n - 1] *= (alpha_ + one);
    }
    return w;
}

}  // namespace quadrature
}  // namespace polycalc