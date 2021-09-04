/**
 * \file       gauss_lobatto.hpp
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

    // Good Decimal Calculator found at following site
    // https://keisan.casio.com/exec/system/1280801905

    // Zeros to return
    std::vector<value_type> x(n);

    switch (n) {
        case 1:
            x[0] = 0.0;
            break;
        case 2:
            x[0] = -1.0;
            x[1] = +1.0;
            break;
        case 3:
            x[0] = -1.0;
            x[1] = 0.0;
            x[2] = +1.0;
            break;
        case 4:
            x[0] = -1.0;
            x[1] = -std::sqrt(5.0L) / 5.0L;
            x[2] = +std::sqrt(5.0L) / 5.0L;
            x[3] = +1.0;
            break;
        case 5:
            x[0] = -1.0;
            x[1] = -std::sqrt(21.0L) / 7.0L;
            x[2] = 0.0;
            x[3] = +std::sqrt(21.0L) / 7.0L;
            x[4] = +1.0;
            break;
        case 6:
            x[0] = -1.0;
            x[1] = -std::sqrt((7.0L+2.0L*std::sqrt(7.0L))/21.0L);
            x[2] = -std::sqrt((7.0L-2.0L*std::sqrt(7.0L))/21.0L);
            x[3] = +std::sqrt((7.0L-2.0L*std::sqrt(7.0L))/21.0L);
            x[4] = +std::sqrt((7.0L+2.0L*std::sqrt(7.0L))/21.0L);
            x[5] = +1.0;
            break;
        default:
            polynomial jac(alpha_ + 1, beta_ + 1);
            auto zeros = jac.zeros(n - 2);

            x.front() = -1.0;
            std::copy(zeros.begin(), zeros.end(), x.begin() + 1);
            x.back() = +1.0;
            if (!(n % 2 == 0)) {
                x[std::ptrdiff_t(n / 2)] = 0;  // Correct 10E-16 error at zero
            }
    }
    return x;
}

template <typename T, typename P>
std::vector<typename GaussLobatto<T, P>::value_type> GaussLobatto<T, P>::weights(const unsigned n) const {
    assert(n > 0);

    // Good Decimal Calculator found at following site
    // https://keisan.casio.com/exec/system/1280801905
    
    // Weights to return
    std::vector<value_type> w(n);

    switch (n) {
        case 1:
            w[0] = +2.0;
            break;
        case 2:
            w[0] = +1.0;
            w[1] = +1.0;
            break;
        case 3:
            w[0] = 1.0L / 3.0L;
            w[1] = 4.0L / 3.0L;
            w[2] = 1.0L / 3.0L;
            break;
        case 4:
            w[0] = 1.0L / 6.0L;
            w[1] = 5.0L / 6.0L;
            w[2] = 5.0L / 6.0L;
            w[3] = 1.0L / 6.0L;
            break;
        case 5:
            w[0] =  1.0L / 10.0L;
            w[1] = 49.0L / 90.0L;
            w[2] = 32.0L / 45.0L;
            w[3] = 49.0L / 90.0L;
            w[4] =  1.0L / 10.0L;
            break;
        case 6:
            w[0] =  1.0L / 15.0L;
            w[1] = (14.0L-std::sqrt(7.0L))/30.0L;
            w[2] = (14.0L+std::sqrt(7.0L))/30.0L;
            w[3] = (14.0L+std::sqrt(7.0L))/30.0L;
            w[4] = (14.0L-std::sqrt(7.0L))/30.0L;
            w[5] =  1.0L / 15.0L;
            break;
        default:

            // Get location of zeros
            auto z = this->zeros(n);

            // Evaluate Jacobi n-1 polynomial at each zero
            polynomial jac(alpha_, beta_);
            for (size_type i = 0; i < n; ++i) {
                w[i] = jac.eval(n - 1, z[i]);
            }

            const value_type one = 1;
            const value_type two = 2;
            const value_type apb = alpha_ + beta_;

            value_type fac;
            fac = std::pow(two, apb + one) * std::tgamma(alpha_ + n) * std::tgamma(beta_ + n);
            fac /= (n - 1) * std::tgamma(n) * std::tgamma(alpha_ + beta_ + n + one);

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