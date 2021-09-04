

#include <cmath>
#include <iomanip>
#include <iostream>

#include "polycalc/quadrature/gauss_lobatto.hpp"


#include "catch.hpp"
#include "utilities.hpp"


TEST_CASE("Jacobi Polynomial", "[default]") {
    using namespace ::polycalc::quadrature;
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

    SECTION("Legendre Expansion N = 3") {
        GaussLobatto<value_type> gll(0,0);

        const int N = 3;
        auto z = gll.zeros(N);
        auto w = gll.weights(N);

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], zero));
        REQUIRE(almost_equal(z[2], +one));

        REQUIRE(almost_equal(w[0], one/three));
        REQUIRE(almost_equal(w[1], four/three));
        REQUIRE(almost_equal(w[2], one/three));
    }

    SECTION("Legendre Expansion N = 4") {
        GaussLobatto<value_type> gll(0,0);

        const int N = 4;
        auto z = gll.zeros(N);
        auto w = gll.weights(N);

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], -std::sqrt(five)/five));
        REQUIRE(almost_equal(z[2], +std::sqrt(five)/five));
        REQUIRE(almost_equal(z[3], +one));

        REQUIRE(almost_equal(w[0], one/six));
        REQUIRE(almost_equal(w[1], five/six));
        REQUIRE(almost_equal(w[2], five/six));
        REQUIRE(almost_equal(w[3], one/six));
    }

    SECTION("Legendre Expansion N = 5") {
        GaussLobatto<value_type> gll(0,0);

        const int N = 5;
        auto z = gll.zeros(N);
        auto w = gll.weights(N);

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], -std::sqrt(21.0)/seven));
        REQUIRE(almost_equal(z[2], zero));
        REQUIRE(almost_equal(z[3], +std::sqrt(21.0)/seven));
        REQUIRE(almost_equal(z[4], +one));

        REQUIRE(almost_equal(w[0], one/ten));
        REQUIRE(almost_equal(w[1], 49./90.));
        REQUIRE(almost_equal(w[2], 32./45.));
        REQUIRE(almost_equal(w[3], 49./90.));
        REQUIRE(almost_equal(w[4], one/ten));
    }

    SECTION("Legendre Expansion N = 7") {
        GaussLobatto<value_type> gll(0,0);

        const int N = 7;
        auto z = gll.zeros(N);
        auto w = gll.weights(N);

        REQUIRE(almost_equal(z[0], -1.0));
        REQUIRE(almost_equal(z[1], -0.830223896278566929872));
        REQUIRE(almost_equal(z[2], -0.4688487934707142138038));
        REQUIRE(almost_equal(z[3], 0.0));
        REQUIRE(almost_equal(z[4], +0.4688487934707142138038));
        REQUIRE(almost_equal(z[5], +0.830223896278566929872));
        REQUIRE(almost_equal(z[6], +1.0));

        REQUIRE(almost_equal(w[0], 0.04761904761904761904762));
        REQUIRE(almost_equal(w[1], 0.2768260473615657)); // Not exact
        REQUIRE(almost_equal(w[2], 0.4317453812098626234179));
        REQUIRE(almost_equal(w[3], 0.487619047619047619048));
        REQUIRE(almost_equal(w[4], 0.4317453812098626234179));
        REQUIRE(almost_equal(w[5], 0.2768260473615657)); // Not exact
        REQUIRE(almost_equal(w[6], 0.04761904761904761904762));
    }

}