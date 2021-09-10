/**
 * \file       gauss_lobatto_chebychev.hpp
 * \author     Bryan Flynt
 * \date       Sep 07, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once


#include "polycalc/parameters.hpp"
#include "polycalc/quadrature/gauss_lobatto_jacobi.hpp"

namespace polycalc {
namespace quadrature {

template<typename Coefficient, typename P = DefaultParameters<Coefficient> >
struct GaussLobattoChebychev {

    /** Quadrature Locations
     *
     * Returns the Gauss-Lobatto-Chebychev quadrature locations at last through first locations.
     */
    template <typename RandomAccessIterator>
    static void zeros(RandomAccessIterator first, RandomAccessIterator last) {
        GaussLobattoJacobi<Coefficient,P>::zeros(alpha,beta,first,last);
    }

    /** Quadrature Locations
     *
     * Returns the Gauss-Lobatto-Chebychev quadrature locations at n locations.
     */
    template <typename Iterator>
    static Iterator zeros_n(const unsigned n, Iterator iter){
        GaussLobattoJacobi<Coefficient,P>::zeros_n(alpha,beta,n,first);
    }

    /** Quadrature Weights
     *
     * Returns the Gauss-Lobatto-Chebychev weights at last-first Lobatto zeros.
     */
    template <typename RandomAccessIterator>
    static void weights(RandomAccessIterator first, RandomAccessIterator last) {
        GaussLobattoJacobi<Coefficient,P>::weights(alpha,beta,first,last);
    }

    /** Quadrature Weights
     *
     * Returns the Gauss-Lobatto-Chebychev weights at n Lobatto zeros.
     */
    template <typename Iterator>
    static Iterator weights_n(const unsigned n, Iterator iter){
        GaussLobattoJacobi<Coefficient,P>::weights_n(alpha,beta,n,first);
    }

private:
    static constexpr Coefficient alpha = -0.5;
    static constexpr Coefficient beta  = -0.5;
};

} /* namespace quadrature */
} /* namespace polycalc */