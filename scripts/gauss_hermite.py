
import numpy as np
import sympy as sp
from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf



def gauss_hermite(n, n_digits):
    r"""
    Computes the Gauss-Hermite quadrature [1]_ points and weights.

    The Gauss-Hermite quadrature approximates the integral:

    .. math::
        \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `H_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2^{n-1} n! \sqrt{\pi}}{n^2 \left(H_{n-1}(x_i)\right)^2}

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_hermite
    >>> x, w = gauss_hermite(3, 5)
    >>> x
    [-1.2247, 0, 1.2247]
    >>> w
    [0.29541, 1.1816, 0.29541]

    >>> x, w = gauss_hermite(6, 5)
    >>> x
    [-2.3506, -1.3358, -0.43608, 0.43608, 1.3358, 2.3506]
    >>> w
    [0.00453, 0.15707, 0.72463, 0.72463, 0.15707, 0.00453]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_gen_laguerre, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss-Hermite_Quadrature
    .. [2] http://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
    .. [3] http://people.sc.fsu.edu/~jburkardt/cpp_src/gen_hermite_rule/gen_hermite_rule.html
    """
    x = Dummy("x")
    p = hermite_poly(n, x, polys=True)
    p1 = hermite_poly(n-1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S(1)/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append(((2**(n-1) * factorial(n) * sqrt(pi)) /
                 (n**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w



digits = 36
npoints = list(range(1,17))

for n in npoints:
    xi, wi = gauss_hermite(n, digits)
    
    print('\nN = %3d'%(n))
    for i in range(0,n):
        print('%3d %40.36f %40.36f'%(i+1,xi[i],wi[i]))
    
    