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
