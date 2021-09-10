

#include <cmath>
#include <iomanip>
#include <iostream>

#include "polycalc/interpolation/lagrange.hpp"


#include "catch.hpp"
#include "utilities.hpp"


TEST_CASE("Jacobi Polynomial", "[default]") {
    using namespace ::polycalc::interpolation;
    using value_type = double;
    using coeff_type = long double;

    SECTION("Lagrange Expansion") {
        Lagrange<coeff_type> lag;

        const int N = 8;
        value_type xstart = -1;
        value_type xend   = +1;
        value_type dx     = (xend - xstart)/N;
        std::vector<value_type> x(N);
        for(auto i = 0; i < N; ++i){
            x[i] = xstart + i*dx;
        }

        lag.reset(x);
        for(auto i = 0; i < N; ++i){
            for(auto j = 0; j < N; ++j){

                if( i == j ){
                    REQUIRE(almost_equal(lag.eval(i, x[j]), 1.0L));
                }
                else {
                    REQUIRE(almost_equal(lag.eval(i, x[j]), 0.0L));
                }
            }
        }
    }

    SECTION("Lagrange Evaluation (Easy)") {
        Lagrange<coeff_type> lag;

        // Known Function
        auto func = [](auto a) {
            return -1.0 + 2.0*a - 3.0*a*a;
        };

        // dfdx of Known Function
        auto dfunc = [](auto a) {
            return 2.0 - 6.0*a;
        };

        const int N = 12;
        value_type xstart = -1;
        value_type xend   = +1;
        value_type dx     = (xend - xstart)/N;
        std::vector<value_type> x(N);
        std::vector<value_type> f(N);
        std::vector<value_type> dfdx(N);
        for(auto i = 0; i < N; ++i){
            x[i] = xstart + i*dx;
            f[i] = func(x[i]);
        }

        // Get Exact Solutions
        value_type x_exact    = 0.5 * (x[1] + x[2]);
        value_type f_exact    = func(x_exact);
        value_type dfdx_exact = dfunc(x_exact);

        // Get Weights for Value and Deriv
        lag.reset(x);
        std::vector<value_type> w(N);
        std::vector<value_type> dw(N);
        for(auto i = 0; i < N; ++i){
            w[i]  = lag.eval(i, x_exact);
            dw[i] = lag.ddx(i, x_exact);
        }

        // Combine
        value_type f_pred = 0;
        value_type dfdx_pred = 0;
        for(auto i = 0; i < N; ++i){
            f_pred    += w[i]  * f[i];
            dfdx_pred += dw[i] * f[i];
        }

        // Interpolation is not almost_equal but close
        REQUIRE(almost_equal(f_pred, f_exact));
        REQUIRE(almost_equal(dfdx_pred, dfdx_exact));

    }

    // Exercises the calc_diff function and exposes 
    // the bad solutions is finds if used
    SECTION("Lagrange Evaluation (Hard)") {
        Lagrange<coeff_type> lag;

        // Known Function
        auto func = [](auto a) {
            return -1.0 + 2.0*a - 3.0*a*a;
        };

        // dfdx of Known Function
        auto dfunc = [](auto a) {
            return 2.0 - 6.0*a;
        };

        const int N = 12;
        value_type xstart = -1;
        value_type xend   = +1;
        value_type dx     = (xend - xstart)/N;
        std::vector<value_type> x(N);
        std::vector<value_type> f(N);
        std::vector<value_type> dfdx(N);
        for(auto i = 0; i < N; ++i){
            x[i] = xstart + i*dx;
            f[i] = -1.0 + 2.0*x[i] - 3*x[i]*x[i];
        }

        // Get Weights for Value and Deriv
        value_type x_exact    = x[N/2] + 1.0E-6L;
        value_type f_exact    = func(x_exact);
        value_type dfdx_exact = dfunc(x_exact);


        lag.reset(x);
        std::vector<value_type> w(N);
        std::vector<value_type> dw(N);
        for(auto i = 0; i < N; ++i){
            w[i]  = lag.eval(i, x_exact);
            dw[i] = lag.ddx(i, x_exact);
        }

        // Combine
        value_type f_pred = 0;
        value_type dfdx_pred = 0;
        for(auto i = 0; i < N; ++i){
            f_pred    += w[i]  * f[i];
            dfdx_pred += dw[i] * f[i];
        }

        // Interpolation is not almost_equal but close
        REQUIRE(almost_equal(f_pred, f_exact));
        REQUIRE(almost_equal(dfdx_pred, dfdx_exact));

    }


}