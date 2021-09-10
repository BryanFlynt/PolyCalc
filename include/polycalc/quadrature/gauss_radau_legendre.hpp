/**
 * \file       gauss_lobatto_chebychev.hpp
 * \author     Bryan Flynt
 * \date       Sep 07, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once


#include "polycalc/parameters.hpp"
#include "polycalc/quadrature/gauss_radau_jacobi.hpp"

namespace polycalc {
namespace quadrature {

template<typename Coefficient, typename P = DefaultParameters<Coefficient> >
struct GaussRadauLegendre {

    /** Quadrature Locations
     *
     * Returns the Gauss-Radau-Legendre quadrature locations at last through first locations.
     */
    template <short Z1, typename RandomAccessIterator>
    static void zeros(RandomAccessIterator first, RandomAccessIterator last) {
        GaussRadauJacobi<Coefficient,P>::zeros<Z1>(alpha,beta,first,last);
    }

    /** Quadrature Locations
     *
     * Returns the Gauss-Radau-Legendre quadrature locations at n locations.
     */
    template <short Z1, typename Iterator>
    static Iterator zeros_n(const unsigned n, Iterator iter){
        GaussRadauJacobi<Coefficient,P>::zeros_n<Z1>(alpha,beta,n,first);
    }

    /** Quadrature Weights
     *
     * Returns the Gauss-Radau-Legendre weights at last-first Radau zeros.
     */
    template <short Z1, typename RandomAccessIterator>
    static void weights(RandomAccessIterator first, RandomAccessIterator last) {
        GaussRadauJacobi<Coefficient,P>::weights<Z1>(alpha,beta,first,last);
    }

    /** Quadrature Weights
     *
     * Returns the Gauss-Radau-Legendre weights at n Radau zeros.
     */
    template <short Z1, typename Iterator>
    static Iterator weights_n(const unsigned n, Iterator iter){
        GaussRadauJacobi<Coefficient,P>::weights_n<Z1>(alpha,beta,n,first);
    }

private:
    static constexpr Coefficient alpha = 0;
    static constexpr Coefficient beta  = 0;
};

} /* namespace quadrature */
} /* namespace polycalc */