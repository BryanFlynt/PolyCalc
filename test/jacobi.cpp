

#include "polycalc/polynomial/jacobi.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "catch.hpp"
#include "utilities.hpp"

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

        constexpr int N = 1;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], zero));
    }

    SECTION("Legendre Roots N = 2") {
        Jacobi<value_type> jac(0, 0);

        constexpr int N = 2;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], -one / std::sqrt(three)));
        REQUIRE(almost_equal(z[1], +one / std::sqrt(three)));
    }

    SECTION("Legendre Roots N = 3") {
        Jacobi<value_type> jac(0, 0);

        constexpr int N = 3;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], -std::sqrt(three / five)));
        REQUIRE(almost_equal(z[1], zero));
        REQUIRE(almost_equal(z[2], +std::sqrt(three / five)));
    }

    SECTION("Legendre Roots N = 4") {
        Jacobi<value_type> jac(0, 0);

        constexpr int N = 4;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], -std::sqrt(three / seven + two / seven * std::sqrt(six / five))));
        REQUIRE(almost_equal(z[1], -std::sqrt(three / seven - two / seven * std::sqrt(six / five))));
        REQUIRE(almost_equal(z[2], +std::sqrt(three / seven - two / seven * std::sqrt(six / five))));
        REQUIRE(almost_equal(z[3], +std::sqrt(three / seven + two / seven * std::sqrt(six / five))));
    }

    SECTION("Legendre Roots N = 5") {
        Jacobi<value_type> jac(0, 0);

        constexpr int N = 5;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], -one / three * std::sqrt(five + two * std::sqrt(ten / seven))));
        REQUIRE(almost_equal(z[1], -one / three * std::sqrt(five - two * std::sqrt(ten / seven))));
        REQUIRE(almost_equal(z[2], zero));
        REQUIRE(almost_equal(z[3], +one / three * std::sqrt(five - two * std::sqrt(ten / seven))));
        REQUIRE(almost_equal(z[4], +one / three * std::sqrt(five + two * std::sqrt(ten / seven))));
    }

    SECTION("Lobatto Legendre Roots N = 19") {
        Jacobi<value_type> jac(1, 1);

        constexpr int N = 19;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], -0.98257229660454802823448127655540587686));
        REQUIRE(almost_equal(z[1], -0.941976296959745534296));
        REQUIRE(almost_equal(z[2], -0.87929475532359046445115359630494404771));
        REQUIRE(almost_equal(z[3], -0.79600192607771240474431258966035863909));
        REQUIRE(almost_equal(z[4], -0.69405102606222323262731639319466662876));
        REQUIRE(almost_equal(z[5], -0.57583196026183068692702187033808528734));
        REQUIRE(almost_equal(z[6], -0.44411578327900210119451634960735128474));
        REQUIRE(almost_equal(z[7], -0.30198985650876488727535186785875223202));
        REQUIRE(almost_equal(z[8], -0.15278551580218546600635832848566943552));
        REQUIRE(almost_equal(z[9], 0.0));
        REQUIRE(almost_equal(z[10], 0.15278551580218546600635832848566943552));
        REQUIRE(almost_equal(z[11], 0.30198985650876488727535186785875223202));
        REQUIRE(almost_equal(z[12], 0.44411578327900210119451634960735128474));
        REQUIRE(almost_equal(z[13], 0.57583196026183068692702187033808528734));
        REQUIRE(almost_equal(z[14], 0.69405102606222323262731639319466662876));
        REQUIRE(almost_equal(z[15], 0.79600192607771240474431258966035863909));
        REQUIRE(almost_equal(z[16], 0.87929475532359046445115359630494404771));
        REQUIRE(almost_equal(z[17], 0.941976296959745534296));
        REQUIRE(almost_equal(z[18], 0.98257229660454802823448127655540587686));
    }

    SECTION("Chebychev Roots N = 1") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int N = 1;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        REQUIRE(almost_equal(z[0], zero));  // cos has error around 0
    }

    SECTION("Chebychev Roots N = 2") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int N = 2;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        for (auto k = 1; k <= N; ++k) {
            REQUIRE(almost_equal(z[k - 1], -std::cos(pi * (2 * k - 1) / (2 * N))));
        }
    }

    SECTION("Chebychev Roots N = 3") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int N = 3;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        for (auto k = 1; k <= N; ++k) {
            if (k == (N + 1) / 2) {
                REQUIRE(almost_equal(z[k - 1], zero));  // cos has error around 0
            } else {
                REQUIRE(almost_equal(z[k - 1], -std::cos(pi * (2 * k - 1) / (2 * N))));
            }
        }
    }

    SECTION("Chebychev Roots N = 4") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int N = 4;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        for (auto k = 1; k <= N; ++k) {
            REQUIRE(almost_equal(z[k - 1], -std::cos(pi * (2 * k - 1) / (2 * N))));
        }
    }

    SECTION("Chebychev Roots N = 5") {
        Jacobi<value_type> jac(-0.5, -0.5);

        constexpr int N = 5;
        std::vector<value_type> z;
        jac.zeros_n(N, std::back_inserter(z));

        for (auto k = 1; k <= N; ++k) {
            if (k == (N + 1) / 2) {
                REQUIRE(almost_equal(z[k - 1], zero));  // cos has error around 0
            } else {
                REQUIRE(almost_equal(z[k - 1], -std::cos(pi * (2 * k - 1) / (2 * N))));
            }
        }
    }
}