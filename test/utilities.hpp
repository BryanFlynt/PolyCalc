/**
 * \file       utlities.hpp
 * \author     Bryan Flynt
 * \date       Sep 03, 2021
 * \copyright  Copyright (C) 2021 Bryan Flynt - All Rights Reserved
 */
#pragma once


template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y) {
    constexpr T eps  = 10 * std::numeric_limits<T>::epsilon();
    constexpr T zero = 0;
    const T abs_diff = std::abs(x - y);

    std::cout << std::setprecision(16);
    std::cout << "x               = " << x << std::endl;
    std::cout << "y               = " << y << std::endl;
    std::cout << "std::abs(x - y) = " << abs_diff << std::endl;
    std::cout << "eps             = " << eps << std::endl;
    std::cout << "eps * abs(x)    = " << eps * std::abs(x) << std::endl;
    std::cout << "eps * abs(y)    = " << eps * std::abs(y) << std::endl;

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

template <typename T>
void display_vector(const std::vector<T>& vec) {
    std::cout << std::setprecision(16);
    for (auto i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << std::endl;
    }
}