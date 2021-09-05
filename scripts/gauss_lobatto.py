import numpy as np
import sympy as sp
from sympy.core import S, Dummy
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf



def gauss_lobatto(n, n_digits):
    r"""
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
    and the weights `w_i` are given by:

    .. math::
        &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
        &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

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

    >>> from sympy.integrals.quadrature import gauss_lobatto
    >>> x, w = gauss_lobatto(3, 5)
    >>> x
    [-1, 0, 1]
    >>> w
    [0.33333, 1.3333, 0.33333]
    >>> x, w = gauss_lobatto(4, 5)
    >>> x
    [-1, -0.44721, 0.44721, 1]
    >>> w
    [0.16667, 0.83333, 0.83333, 0.16667]

    See Also
    ========

    gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    .. [2] http://people.math.sfu.ca/~cbm/aands/page_888.htm
    """
    x = Dummy("x")
    p = legendre_poly(n-1, x, polys=True)
    pd = p.diff(x)
    xi = []
    wi = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S(1)/10**(n_digits+2))
        xi.append(r.n(n_digits))
        wi.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    wi.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    wi.append((S(2)/(n*(n-1))).n(n_digits))
    return xi, wi


digits = 36
npoints = list(range(2,17))

for n in npoints:
    xi, wi = gauss_lobatto(n, digits)
    
    print('\nN = %3d'%(n))
    for i in range(0,n):
        print('%3d %40.36f %40.36f'%(i+1,xi[i],wi[i]))
    
    
    
    
    
