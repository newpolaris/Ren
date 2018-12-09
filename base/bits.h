#pragma once

// from boost
template <class integral, class size_t>
constexpr integral align_up(integral x, size_t a) noexcept {
    return integral((x + (integral(a) - 1)) & ~integral(a - 1));
}

template <class integral, class size_t>
constexpr bool bit_test(integral x, size_t bit) noexcept {
    return x & (1 << bit);
}

template <class integral_1, class integral_2>
bool flag_test(integral_1 x, integral_2 flag) noexcept {
    return (x & flag) == flag;
}

