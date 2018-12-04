#pragma once

#include "volk.h"
#include "spirv-reflect.h"

struct ShaderModule
{   
    VkShaderModule module;
    SpirvReflections reflections;
};

ShaderModule CreateShaderModule(VkDevice device, const char* filename);
