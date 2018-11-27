#pragma once

#include <cassert>
#include <volk.h>
#include <vector>
#include <stdint.h>

#ifndef VK_CECHK
#define VK_CHECK(call) \
    do { \
        VkResult result_ = call; \
        assert(result_ == VK_SUCCESS); \
    } while (0)
#endif

#ifndef ARRAYSIZE
#define ARRAYSIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

