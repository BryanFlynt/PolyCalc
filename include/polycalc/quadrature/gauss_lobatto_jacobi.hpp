/**
 * \file       gauss_lobatto_jacobi.hpp
 * \author     Bryan Flynt
 * \date       Sep 07, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once


#include <algorithm>
#include <cassert>


#include "polycalc/parameters.hpp"
#include "polycalc/polynomial/jacobi.hpp"

namespace polycalc {
namespace quadrature {

/**
 * Function for Gauss-Lobatto-Jacobi roots and weights
 * 
 * Function to calculate the integration locations and weights
 * for the Gauss Lobatto Jacobi polynomials.  
 * 
 * @tparam Coefficient floating point type used to perform internal calculations
 */
template<typename Coefficient, typename P = DefaultParameters<Coefficient> >
struct GaussLobattoJacobi {

    using size_type  = std::size_t;
    using coeff_type = Coefficient;
    using params     = P;
    using polynomial = ::polycalc::polynomial::Jacobi<Coefficient, P>;

    /** Quadrature Locations
     *
     * Returns the Gauss-Lobatto-Jacobi quadrature locations at last through first locations.
     */
    template <typename RandomAccessIterator>
    static void zeros(const Coefficient alpha, const Coefficient beta, RandomAccessIterator first, RandomAccessIterator last);

    /** Quadrature Locations
     *
     * Returns the Gauss-Lobatto-Jacobi quadrature locations at n locations.
     */
    template <typename Iterator>
    static Iterator zeros_n(const Coefficient alpha, const Coefficient beta, const unsigned n, Iterator iter);

    /** Quadrature Weights
     *
     * Returns the Gauss-Lobatto-Jacobi weights at last-first Lobatto zeros.
     */
    template <typename RandomAccessIterator>
    static void weights(const Coefficient alpha, const Coefficient beta, RandomAccessIterator first, RandomAccessIterator last);

    /** Quadrature Weights
     *
     * Returns the Gauss-Lobatto-Jacobi weights at n Lobatto zeros.
     */
    template <typename Iterator>
    static Iterator weights_n(const Coefficient alpha, const Coefficient beta, const unsigned n, Iterator iter);

};



template<typename C, typename P>
template <typename RandomAccessIterator>
void GaussLobattoJacobi<C,P>::zeros(const C alpha, const C beta, RandomAccessIterator first, RandomAccessIterator last) {
    GaussLobattoJacobi<C,P>::zeros_n(alpha,beta,std::distance(first, last), first);
}

template<typename C, typename P>
template <typename Iterator>
Iterator GaussLobattoJacobi<C,P>::zeros_n(const C alpha, const C beta, const unsigned n, Iterator iter) {
    assert(n > 1);

    switch (n) {
        case 2:
            *iter++ = -1;
            *iter++ = 1;
            break;
        case 3:
            *iter++ = -1;
            *iter++ = 0;
            *iter++ = 1;
            break;
        default:
            polynomial jac(alpha + 1, beta + 1);
            *iter++ = -1;
            iter    = jac.zeros_n(n - 2, iter);
            *iter++ = 1;
    }
    return iter;
}

/** Quadrature Weights
 *
 * Returns the Gauss-Jacobi weights at n Lobatto zeros.
 */
template<typename C, typename P>
template <typename RandomAccessIterator>
void GaussLobattoJacobi<C,P>::weights(const C alpha, const C beta, RandomAccessIterator first, RandomAccessIterator last) {
    GaussLobattoJacobi<C,P>::weights_n(alpha,beta,std::distance(first, last), first);
}

/** Quadrature Weights
 *
 * Returns the Gauss-Jacobi weights at n Lobatto zeros.
 */
template<typename C, typename P>
template <typename Iterator>
Iterator GaussLobattoJacobi<C,P>::weights_n(const C alpha, const C beta, const unsigned n, Iterator iter) {
    assert(n > 1);

    switch (n) {
        case 2:
            *iter++ = 1.0;
            *iter++ = 1.0;
            break;
        case 3:
            *iter++ = 0.33333333333333333333333333333333333333;
            *iter++ = 1.3333333333333333333333333333333333333;
            *iter++ = 0.33333333333333333333333333333333333333;
            break;
        default:
            using coeff_type = long double; // Highest Precision

            // Local memory for calcs
            coeff_type z[n];  // Zero points
            coeff_type w[n];  // local weight calcs

            // Get location of zeros
            GaussLobattoJacobi<C,P>::zeros_n(alpha, beta, n, z);

            // Evaluate Jacobi n-1 polynomial at each zero
            polynomial jac(alpha, beta);
            for (std::size_t i = 0; i < n; ++i) {
                w[i] = jac.eval(n - 1, z[i]);
            }

            const coeff_type one = 1;
            const coeff_type two = 2;
            const coeff_type apb = alpha + beta;

            coeff_type fac;
            fac = std::pow(two, apb + one) * std::tgamma(alpha + n) * std::tgamma(beta + n);
            fac /= (n - 1) * std::tgamma(n) * std::tgamma(alpha + beta + n + one);

            for (std::size_t i = 0; i < n; ++i) {
                w[i] = fac / (w[i] * w[i]);
            }
            w[0] *= (beta + one);
            w[n - 1] *= (alpha + one);
            iter = std::copy(w, w + n, iter);
    }
    return iter;
}

} /* namespace quadrature */
} /* namespace polycalc */
