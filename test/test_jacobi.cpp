

#include "polycalc/polynomial/jacobi.hpp"

#include <iomanip>
#include <iostream>


int main() {
    using namespace ::polycalc::polynomial;
    using Real = double;

    Jacobi<Real> jac(0,0);

    std::cout << jac.eval(0, 0.0) << std::endl;
    std::cout << jac.ddx(0, 0.0)  << std::endl;

    std::cout << jac.eval(1, 0.0) << std::endl;
    std::cout << jac.ddx(1, 0.0)  << std::endl;

    std::cout << jac.eval(2, 0.0) << std::endl;
    std::cout << jac.ddx(2, 0.0)  << std::endl;

    auto z = jac.zeros(7);
    std::cout << std::setprecision(16);
    for(auto i = 0; i < z.size(); ++i){
        std::cout << z[i] << std::endl;
    }

    return 0;
}