#pragma once

#include "volk.h"
#include "spirv-reflect.h"

struct ShaderModule
{   
    VkShaderModule module;
    spirv::SpirvReflections reflections;
};

struct InputInterfaceAttribute
{
    VkFormat format;
    uint32_t location;
    uint32_t stride;
    std::string name;
};

using VariableRef = std::reference_wrapper<const spirv::Variable>;
using VariableRefList = std::vector<VariableRef>;
using InputInterfaceAttributeList = std::vector<InputInterfaceAttribute>;

ShaderModule CreateShaderModule(VkDevice device, const char* filename);

VkPipelineShaderStageCreateInfo GetShaderStageCreateInfo(const ShaderModule& shader, const std::string& name);

InputInterfaceAttributeList GetInputInterfaceVariables(const ShaderModule& shader, const std::string& name);
VariableRefList GetInterfaceVariableReferences(const ShaderModule& shader, const std::string& name);
std::optional<InputInterfaceAttribute> FindVariable(const InputInterfaceAttributeList& list, const std::string& name);
