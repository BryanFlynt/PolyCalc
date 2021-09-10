

#include "polycalc/quadrature/gauss_lobatto_legendre.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>

#include "catch.hpp"
#include "utilities.hpp"

TEST_CASE("GLL", "[default]") {
    using namespace ::polycalc::quadrature;
    using coef_type  = long double;
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
        const int N = 3;
        std::vector<value_type> z;
        std::vector<value_type> w;
        GaussLobattoLegendre<coef_type>::zeros_n(N, std::back_inserter(z));
        GaussLobattoLegendre<coef_type>::weights_n(N, std::back_inserter(w));

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], zero));
        REQUIRE(almost_equal(z[2], +one));

        REQUIRE(almost_equal(w[0], one / three));
        REQUIRE(almost_equal(w[1], four / three));
        REQUIRE(almost_equal(w[2], one / three));
    }

    SECTION("Legendre Expansion N = 4") {
        const int N = 4;
        std::vector<value_type> z;
        std::vector<value_type> w;
        GaussLobattoLegendre<coef_type>::zeros_n(N, std::back_inserter(z));
        GaussLobattoLegendre<coef_type>::weights_n(N, std::back_inserter(w));

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], -std::sqrt(five) / five));
        REQUIRE(almost_equal(z[2], +std::sqrt(five) / five));
        REQUIRE(almost_equal(z[3], +one));

        REQUIRE(almost_equal(w[0], one / six));
        REQUIRE(almost_equal(w[1], five / six));
        REQUIRE(almost_equal(w[2], five / six));
        REQUIRE(almost_equal(w[3], one / six));
    }

    SECTION("Legendre Expansion N = 5") {
        const int N = 5;
        std::vector<value_type> z;
        std::vector<value_type> w;
        GaussLobattoLegendre<coef_type>::zeros_n(N, std::back_inserter(z));
        GaussLobattoLegendre<coef_type>::weights_n(N, std::back_inserter(w));

        REQUIRE(almost_equal(z[0], -one));
        REQUIRE(almost_equal(z[1], -std::sqrt(21.0) / seven));
        REQUIRE(almost_equal(z[2], zero));
        REQUIRE(almost_equal(z[3], +std::sqrt(21.0) / seven));
        REQUIRE(almost_equal(z[4], +one));

        REQUIRE(almost_equal(w[0], one / ten));
        REQUIRE(almost_equal(w[1], 49. / 90.));
        REQUIRE(almost_equal(w[2], 32. / 45.));
        REQUIRE(almost_equal(w[3], 49. / 90.));
        REQUIRE(almost_equal(w[4], one / ten));
    }

    SECTION("Legendre Expansion N = 7") {

        std::cout << "HERE" <<std::endl;
        const int N = 7;
        std::vector<value_type> z;
        std::vector<value_type> w;
        GaussLobattoLegendre<coef_type>::zeros_n(N, std::back_inserter(z));
        GaussLobattoLegendre<coef_type>::weights_n(N, std::back_inserter(w));

        REQUIRE(almost_equal(z[0], -1.0));
        REQUIRE(almost_equal(z[1], -0.830223896278566929872));
        REQUIRE(almost_equal(z[2], -0.4688487934707142138038));
        REQUIRE(almost_equal(z[3], 0.0));
        REQUIRE(almost_equal(z[4], +0.4688487934707142138038));
        REQUIRE(almost_equal(z[5], +0.830223896278566929872));
        REQUIRE(almost_equal(z[6], +1.0));

        REQUIRE(almost_equal(w[0], 0.04761904761904761904762));
        REQUIRE(almost_equal(w[1], 0.2768260473615659480107));
        REQUIRE(almost_equal(w[2], 0.4317453812098626234179));
        REQUIRE(almost_equal(w[3], 0.487619047619047619048));
        REQUIRE(almost_equal(w[4], 0.4317453812098626234179));
        REQUIRE(almost_equal(w[5], 0.2768260473615659480107));
        REQUIRE(almost_equal(w[6], 0.04761904761904761904762));
    }

    SECTION("Legendre Expansion N = 21") {
        const int N = 21;
        std::vector<value_type> z;
        std::vector<value_type> w;
        GaussLobattoLegendre<coef_type>::zeros_n(N, std::back_inserter(z));
        GaussLobattoLegendre<coef_type>::weights_n(N, std::back_inserter(w));

        REQUIRE(almost_equal(z[0], -1.0));
        REQUIRE(almost_equal(z[1], -0.982572296604548028234));
        REQUIRE(almost_equal(z[2], -0.941976296959745534296));
        REQUIRE(almost_equal(z[3], -0.87929475532359046445115359630494404771));
        REQUIRE(almost_equal(z[4], -0.79600192607771240474431258966035863909));
        REQUIRE(almost_equal(z[5], -0.69405102606222323262731639319466662876));
        REQUIRE(almost_equal(z[6], -0.57583196026183068692702187033808528734));
        REQUIRE(almost_equal(z[7], -0.44411578327900210119451634960735128474));
        REQUIRE(almost_equal(z[8], -0.30198985650876488727535186785875223202));
        REQUIRE(almost_equal(z[9], -0.15278551580218546600635832848566943552));
        REQUIRE(almost_equal(z[10], 0.0));
        REQUIRE(almost_equal(z[11], 0.15278551580218546600635832848566943552));
        REQUIRE(almost_equal(z[12], 0.30198985650876488727535186785875223202)); 
        REQUIRE(almost_equal(z[13], 0.44411578327900210119451634960735128474)); 
        REQUIRE(almost_equal(z[14], 0.57583196026183068692702187033808528734)); 
        REQUIRE(almost_equal(z[15], 0.69405102606222323262731639319466662876)); 
        REQUIRE(almost_equal(z[16], 0.79600192607771240474431258966035863909)); 
        REQUIRE(almost_equal(z[17], 0.87929475532359046445115359630494404771)); 
        REQUIRE(almost_equal(z[18], 0.941976296959745534296)); 
        REQUIRE(almost_equal(z[19], 0.982572296604548028234)); 
        REQUIRE(almost_equal(z[20], +1.0)); 
    }
}