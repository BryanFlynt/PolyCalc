

#include "polycalc/polynomial/jacobi.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

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
    using namespace ::polycalc::polynomial;
    using value_type = double;

    constexpr value_type zero(0);
    constexpr value_type one(1);
    constexpr value_type two(2);
    constexpr value_type three(3);
    constexpr value_type four(4);
    constexpr value_type five(5);
    constexpr value_type six(6);
    constexpr value_type seven(7);
    constexpr value_type eight(8);
    constexpr value_type nine(9);
    constexpr value_type ten(10);
    constexpr value_type pi = 3.141592653589793238462643383279502884L;

    SECTION("Legendre Roots N = 1") {
        Jacobi<value_type> jac(0, 0);

        auto z = jac.zeros(1);
        REQUIRE(almost_equal(z[0], zero));
    }

    SECTION("Legendre Roots N = 2") {
        Jacobi<value_type> jac(0, 0);

        auto z = jac.zeros(2);
        REQUIRE(almost_equal(z[0], -one / std::sqrt(three)));
        REQUIRE(almost_equal(z[1], +one / std::sqrt(three)));
    }

    SECTION("Legendre Roots N = 3") {
        Jacobi<value_type> jac(0, 0);

        auto z = jac.zeros(3);
        REQUIRE(almost_equal(z[0], -std::sqrt(three/five)));
        REQUIRE(almost_equal(z[1], zero));
        REQUIRE(almost_equal(z[2], +std::sqrt(three/five)));
    }

    SECTION("Legendre Roots N = 4") {
        Jacobi<value_type> jac(0, 0);

        auto z = jac.zeros(4);
        REQUIRE(almost_equal(z[0], -std::sqrt(three/seven + two/seven*std::sqrt(six/five))));
        REQUIRE(almost_equal(z[1], -std::sqrt(three/seven - two/seven*std::sqrt(six/five))));
        REQUIRE(almost_equal(z[2], +std::sqrt(three/seven - two/seven*std::sqrt(six/five))));
        REQUIRE(almost_equal(z[3], +std::sqrt(three/seven + two/seven*std::sqrt(six/five))));
    }

    SECTION("Legendre Roots N = 5") {
        Jacobi<value_type> jac(0, 0);

        auto z = jac.zeros(5);
        REQUIRE(almost_equal(z[0], -one/three*std::sqrt(five + two*std::sqrt(ten/seven))));
        REQUIRE(almost_equal(z[1], -one/three*std::sqrt(five - two*std::sqrt(ten/seven))));
        REQUIRE(almost_equal(z[2], zero));
        REQUIRE(almost_equal(z[3], +one/three*std::sqrt(five - two*std::sqrt(ten/seven))));
        REQUIRE(almost_equal(z[4], +one/three*std::sqrt(five + two*std::sqrt(ten/seven))));
    }


    SECTION("Chebychev Roots N = 1") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int n = 1;
        auto z = jac.zeros(n);
        REQUIRE(almost_equal(z[0], zero) ); // cos has error around 0
    }

    SECTION("Chebychev Roots N = 2") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int n = 2;
        auto z = jac.zeros(n);
        for(auto k = 1; k <= n; ++k){
            REQUIRE(almost_equal(z[k-1], -std::cos(pi*(2*k-1)/(2*n))) );
        }
    }

    SECTION("Chebychev Roots N = 3") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int n = 3;
        auto z = jac.zeros(n);
        for(auto k = 1; k <= n; ++k){
            if( k == (n+1)/2 ){
                REQUIRE(almost_equal(z[k-1], zero) ); // cos has error around 0
            }
            else {
                REQUIRE(almost_equal(z[k-1], -std::cos(pi*(2*k-1)/(2*n))) );
            }
        }
    }

    SECTION("Chebychev Roots N = 4") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int n = 4;
        auto z = jac.zeros(n);
        for(auto k = 1; k <= n; ++k){
            REQUIRE(almost_equal(z[k-1], -std::cos(pi*(2*k-1)/(2*n))) );
        }
    }

    SECTION("Chebychev Roots N = 5") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int n = 5;
        auto z = jac.zeros(n);
        for(auto k = 1; k <= n; ++k){
            if( k == (n+1)/2 ){
                REQUIRE(almost_equal(z[k-1], zero) ); // cos has error around 0
            }
            else {
                REQUIRE(almost_equal(z[k-1], -std::cos(pi*(2*k-1)/(2*n))) );
            }
        }
    }

}