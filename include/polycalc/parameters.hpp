/**
 * \file       parameters.hpp
 * \author     Bryan Flynt
 * \date       Sep 02, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once

#include <limits>

namespace polycalc {

template<typename T>
struct DefaultParameters {
	static constexpr unsigned MAX_ITER = 30;
	static constexpr T TOL = 100*std::numeric_limits<T>::epsilon();
};







} /** polycalc **/