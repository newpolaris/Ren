#pragma once

#include "common.h"

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
