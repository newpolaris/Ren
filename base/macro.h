#pragma once

#ifndef ASSERT
#define ASSERT(x) \
    do { \
        if (!(x)) __assert(#x, 0, __FILE__, __LINE__); \
    } while(0)
#endif
#ifndef __assert
#define __assert(e, code, file, line) \
	((void)printf("%s:%u: failed assertion '%s''%d'\n", file, line, e, code), abort())
#endif

#ifndef VK_ASSERT
#define VK_ASSERT(x) \
    do { \
        VkResult result = x; \
        if ((result != VK_SUCCESS)) __assert(#x, result, __FILE__, __LINE__); \
    } while(0)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif
