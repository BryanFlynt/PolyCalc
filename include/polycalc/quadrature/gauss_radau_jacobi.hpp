/**
 * \file       gauss_radau_jacobi.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <algorithm>
#include <cassert>


#include "polycalc/parameters.hpp"
#include "polycalc/polynomial/jacobi.hpp"

namespace polycalc {
namespace quadrature {

template <typename Coefficient, typename P = DefaultParameters<Coefficient> >
struct GaussRadauJacobi {
    using size_type  = std::size_t;
    using coeff_type = Coefficient;
    using params     = P;
    using polynomial = ::polycalc::polynomial::Jacobi<T, P>;

    /** Quadrature Locations
     *
     * Returns the Gauss-Radau-Jacobi quadrature locations at last through first locations.
     */
    template <short Z1, typename RandomAccessIterator>
    static void zeros(const Coefficient alpha, const Coefficient beta, RandomAccessIterator first,
                      RandomAccessIterator last);

    /** Quadrature Locations
     *
     * Returns the Gauss-Radau-Jacobi quadrature locations at n locations.
     */
    template <short Z1, typename Iterator>
    static Iterator zeros_n(const Coefficient alpha, const Coefficient beta, const unsigned n, Iterator iter);

    /** Quadrature Weights
     *
     * Returns the Gauss-Radau-Jacobi weights at last-first Lobatto zeros.
     */
    template <short Z1, typename RandomAccessIterator>
    static void weights(const Coefficient alpha, const Coefficient beta, RandomAccessIterator first,
                        RandomAccessIterator last);

    /** Quadrature Weights
     *
     * Returns the Gauss-Radau-Jacobi weights at n Lobatto zeros.
     */
    template <short Z1, typename Iterator>
    static Iterator weights_n(const Coefficient alpha, const Coefficient beta, const unsigned n, Iterator iter);
};

template <typename C, typename P>
template <short Z1, typename RandomAccessIterator>
void GaussRadauJacobi<C, P>::zeros(const C alpha, const C beta, RandomAccessIterator first, RandomAccessIterator last) {
    GaussRadauJacobi<C, P>::zeros_n<Z1>(alpha, beta, std::distance(first, last), first);
}

template <typename C, typename P>
template <short Z1, typename Iterator>
Iterator GaussRadauJacobi<C, P>::zeros_n(const C alpha, const C beta, const unsigned n, Iterator iter) {
    if (n == 1) {
        *iter++ = 0;
    } else {
        if constexpr (Z1 > 0) {  // Z = +1
            polynomial jac(alpha + 1, beta);
            iter    = jac.zeros_n(n - 1, iter);
            *iter++ = 1;
        } else if constexpr (Z1 < 0) {  // Z = -1
            polynomial jac(alpha, beta + 1);
            *iter++ = -1;
            iter    = jac.zeros_n(n - 1, iter);
        }
    }
    return iter;
}

/** Quadrature Weights
 *
 * Returns the Gauss-Jacobi weights at n Lobatto zeros.
 */
template <typename C, typename P>
template <short Z1, typename RandomAccessIterator>
void GaussRadauJacobi<C, P>::weights(const C alpha, const C beta, RandomAccessIterator first,
                                     RandomAccessIterator last) {
    GaussRadauJacobi<C, P>::weights_n<Z1>(alpha, beta, std::distance(first, last), first);
}

/** Quadrature Weights
 *
 * Returns the Gauss-Jacobi weights at n Lobatto zeros.
 */
template <typename C, typename P>
template <short Z1, typename Iterator>
Iterator GaussRadauJacobi<C, P>::weights_n(const C alpha, const C beta, const unsigned n, Iterator iter) {
    if (n == 1) {
        *iter++ = 2;
    } else {
        // Local memory for calcs
        coeff_type z[n];  // Zero points
        coeff_type w[n];  // local weight calcs

        // Get location of zeros
        GaussRadauJacobi<C, P>::zeros_n<Z1>(alpha, beta, n, z);

        // Evaluate Jacobi n-1 polynomial at each zero
        polynomial jac(alpha, beta);
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = jac.eval(n - 1, z[i]);
        }

        const coeff_type one = 1;
        const coeff_type two = 2;
        const coeff_type apb = alpha + beta;
        const coeff_type np1 = 1 + n;
        coeff_type fac;

        if constexpr (Z1 > 0) {  // Z = +1
            fac = std::pow(two, apb) * std::tgamma(alpha + n) * std::tgamma(beta + n);
            fac /= std::tgamma(n) * (alpha + n) * std::tgamma(apb + n + 1);

            for (size_type i = 0; i < n; ++i) {
                w[i] = fac * (one + z[i]) / (w[i] * w[i]);
            }
            w[n - 1] *= (alpha + one);
        } else if constexpr (Z1 < 0) {  // Z = -1
            fac = std::pow(two, apb) * std::tgamma(alpha + n) * std::tgamma(beta + n);
            fac /= std::tgamma(n) * (beta + n) * std::tgamma(apb + n + 1);

            for (size_type i = 0; i < n; ++i) {
                w[i] = fac * (one - z[i]) / (w[i] * w[i]);
            }
            w[0] *= (beta + one);
        }
        iter = std::copy(w, w + n, iter);
    }
    return iter;
}

}  // namespace quadrature
}  // namespace polycalc
