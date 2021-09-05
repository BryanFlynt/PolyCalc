import numpy as np
import sympy as sp
from sympy.core import S, Dummy
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf



def symbolic_gauss_legendre(n):
    """
    Computes the symbolic Gauss-Legendre quadrature points and weights.

    Parameters
    ----------
    n : Integer
        Number of integration points.
    x : Symbol
        Symbolic sysmbol.

    Returns
    -------
    xi : Abscisca
        Symbolic location of roots.
    wi : Weights
        Symbolic weights of roots.

    """
    x = Dummy("x")
    Pnx = sp.legendre(n,x)
    Pp = sp.diff(Pnx,x)
    xi = sp.solve( Pnx, x )
    wi = [ sp.simplify(2/(1 - xj**2)/(Pp.subs(x,xj))**2) for xj in xi ]
    return xi, wi


def gauss_legendre(n, n_digits):
    r"""
    Computes the Gauss-Legendre quadrature [1]_ points and weights.

    The Gauss-Legendre quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2}{\left(1-x_i^2\right) \left(P'_n(x_i)\right)^2}

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

    >>> from sympy.integrals.quadrature import gauss_legendre
    >>> x, w = gauss_legendre(3, 5)
    >>> x
    [-0.7746, 0, 0.7746]
    >>> w
    [0.55556, 0.88889, 0.55556]
    >>> x, w = gauss_legendre(4, 5)
    >>> x
    [-0.86114, -0.33998, 0.33998, 0.86114]
    >>> w
    [0.34785, 0.65215, 0.65215, 0.34785]

    See Also
    ========

    gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] http://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule/legendre_rule.html
    """
    x = Dummy("x")
    p = legendre_poly(n, x, polys=True)
    pd = p.diff(x)
    xi = []
    wi = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S(1)/10**(n_digits+2))
        xi.append(r.n(n_digits))
        wi.append((2/((1-r**2) * pd.subs(x, r)**2)).n(n_digits))
    return xi, wi


digits = 36
npoints = list(range(1,17))

for n in npoints:
    xi, wi = gauss_legendre(n, digits)
    
    print('\nN = %3d'%(n))
    for i in range(0,n):
        print('%3d %40.36f %40.36f'%(i+1,xi[i],wi[i]))
    
    
    
    
    
