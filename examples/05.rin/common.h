#pragma once

#ifndef ASSERT
#define ASSERT(x) \
    do { \
        if (!(x)) __assert(#x, __FILE__, __LINE__); \
    } while(0)
#endif
#ifndef __assert
#define __assert(e, file, line) \
	((void)printf("%s:%u: failed assertion `%s'\n", file, line, e), abort())
#endif

#ifndef VK_ASSERT
#define VK_ASSERT(x) \
    do { \
        VkResult result = x; \
        ASSERT(VK_SUCCESS == result); \
    } while(0)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

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
