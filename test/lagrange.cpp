

#include <cmath>
#include <iomanip>
#include <iostream>

#include "polycalc/interpolation/lagrange.hpp"


#include "catch.hpp"

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y) {
    constexpr T eps  = std::numeric_limits<T>::epsilon();
    constexpr T zero = 0;
    const T abs_diff = std::abs(x - y);

    // std::cout << std::setprecision(16);
    // std::cout << "x               = " << x << std::endl;
    // std::cout << "y               = " << y << std::endl;
    // std::cout << "std::abs(x - y) = " << abs_diff << std::endl;
    // std::cout << "eps             = " << eps << std::endl;
    // std::cout << "eps * abs(x)    = " << eps * std::abs(x) << std::endl;
    // std::cout << "eps * abs(y)    = " << eps * std::abs(y) << std::endl;

    if ((x == zero) || (y == zero)) {
        if (abs_diff <= eps) {
            return true;
        }
    } else {
        if ((abs_diff <= (eps * std::abs(x))) || (abs_diff <= (eps * std::abs(y)))) {
            return true;
        }
    }
    return false;
}

// template <typename T>
// void display_vector(const std::vector<T>& vec) {
//     std::cout << std::setprecision(16);
//     for (auto i = 0; i < vec.size(); ++i) {
//         std::cout << vec[i] << std::endl;
//     }
// }

TEST_CASE("Jacobi Polynomial", "[default]") {
    using namespace ::polycalc::interpolation;
    using value_type = double;


    SECTION("Lagrange Expansion") {
        Lagrange<value_type> lag;

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
                    REQUIRE(almost_equal(lag.eval(i, x[j]), 1.0));
                }
                else {
                    REQUIRE(almost_equal(lag.eval(i, x[j]), 0.0));
                }
            }
        }
    }

    SECTION("Lagrange Evaluation (Easy)") {
        Lagrange<value_type> lag;

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
        value_type x_test    = 0.5 * (x[1] + x[2]);
        value_type f_test    = -1.0 + 2.0*x_test - 3*x_test*x_test;
        value_type dfdx_test = 2.0 - 6*x_test;

        lag.reset(x);
        std::vector<value_type> w(N);
        std::vector<value_type> dw(N);
        for(auto i = 0; i < N; ++i){
            w[i]  = lag.eval(i, x_test);
            dw[i] = lag.ddx(i, x_test);
        }

        // Combine
        value_type f_pred = 0;
        value_type dfdx_pred = 0;
        for(auto i = 0; i < N; ++i){
            f_pred    += w[i]  * f[i];
            dfdx_pred += dw[i] * f[i];
        }

        // Interpolation is not almost_equal but close
        REQUIRE(std::abs(f_pred - f_test) < 1.0E-14L );
        REQUIRE( std::abs(dfdx_pred - dfdx_test) < 1.0E-14L );

    }

    // Exercises the calc_diff function and exposes 
    // the bad solutions is finds if used
    SECTION("Lagrange Evaluation (Hard)") {
        Lagrange<value_type> lag;

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
        value_type x_test    = x[N/2] + 1.0E-6L;
        value_type f_test    = -1.0 + 2.0*x_test - 3*x_test*x_test;
        value_type dfdx_test = 2.0 - 6*x_test;

        lag.reset(x);
        std::vector<value_type> w(N);
        std::vector<value_type> dw(N);
        for(auto i = 0; i < N; ++i){
            w[i]  = lag.eval(i, x_test);
            dw[i] = lag.ddx(i, x_test);
        }

        // Combine
        value_type f_pred = 0;
        value_type dfdx_pred = 0;
        for(auto i = 0; i < N; ++i){
            f_pred    += w[i]  * f[i];
            dfdx_pred += dw[i] * f[i];
        }

        // Interpolation is not almost_equal but close
        REQUIRE(std::abs(f_pred - f_test) < 1.0E-14L );
        REQUIRE( std::abs(dfdx_pred - dfdx_test) < 1.0E-14L );

    }


}