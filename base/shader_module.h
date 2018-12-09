#pragma once

#include <volk.h>
#include "spirv-reflect.h"

struct ShaderModule
{   
    VkShaderModule module;
    spirv::SpirvReflections reflections;
};

ShaderModule CreateShaderModule(VkDevice device, const char* filename);

