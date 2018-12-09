#include "shader_module.h"
#include "filesystem.h"
#include "macro.h"

#include <algorithm>

ShaderModule CreateShaderModule(VkDevice device, const char* filepath) {
    auto code = FileRead(filepath);
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateShaderModule(device, &info, nullptr, &module));

    spirv::SpirvReflections reflections = spirv::ReflectShader(code.data(), code.size());
    
    return ShaderModule {
        module,
        std::move(reflections)
    };
}

VkShaderStageFlagBits GetShaderStageBit(SpvExecutionModel model)
{
    switch (model)
    {
        case SpvExecutionModelVertex:
            return VK_SHADER_STAGE_VERTEX_BIT;
        case SpvExecutionModelFragment:
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        case SpvExecutionModelGLCompute:
            return VK_SHADER_STAGE_COMPUTE_BIT;
        case SpvExecutionModelTaskNV:
            return VK_SHADER_STAGE_TASK_BIT_NV;
        case SpvExecutionModelMeshNV:
            return VK_SHADER_STAGE_MESH_BIT_NV;
    }
    ASSERT(FALSE);
}

uint32_t GetPrimitiveStride(const spirv::PrimitiveType& type) {
    return type.width.value() * type.component_count / 8;
}

uint32_t GetPrimitiveStride(const spirv::SpirvReflections& reflections, const uint32_t& type_id) {
    const auto& primitive = reflections.primitive_types.at(type_id);
    return GetPrimitiveStride(primitive);
}

uint32_t GetStride(const spirv::SpirvReflections& reflections, const uint32_t& type_id) {
    auto it = reflections.struct_types.find(type_id);
    if (it != reflections.struct_types.end()) {
        const auto& last = it->second.members.back();
        return last.offset + GetStride(reflections, last.type_id);
    } else {
        auto pit = reflections.primitive_types.find(type_id);
        ASSERT(pit != reflections.primitive_types.end());
        return GetPrimitiveStride(pit->second);
    }
}

VkFormat GetPrimitiveFormat(const spirv::PrimitiveType& type) {
    switch (type.primitive_type) {
    case SpvOpTypeInt:
        switch (type.width.value()) {
        case 8:
            switch (type.component_count) {
            case 1: return type.signedness ? VK_FORMAT_R8_SINT : VK_FORMAT_R8_UINT;
            case 2: return type.signedness ? VK_FORMAT_R8G8_SINT : VK_FORMAT_R8G8_UINT;
            case 3: return type.signedness ? VK_FORMAT_R8G8B8_SINT : VK_FORMAT_R8G8B8_UINT;
            case 4: return type.signedness ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT;
            }
        case 16:
            switch (type.component_count) {
            case 1: return type.signedness ? VK_FORMAT_R16_SINT : VK_FORMAT_R16_UINT;
            case 2: return type.signedness ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_UINT;
            case 3: return type.signedness ? VK_FORMAT_R16G16B16_SINT : VK_FORMAT_R16G16B16_UINT;
            case 4: return type.signedness ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_UINT;
            }
        case 32:
            switch (type.component_count) {
            case 1: return type.signedness ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
            case 2: return type.signedness ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
            case 3: return type.signedness ? VK_FORMAT_R32G32B32_SINT : VK_FORMAT_R32G32B32_UINT;
            case 4: return type.signedness ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
            }
        }
    case SpvOpTypeFloat:
        switch (type.width.value()) {
        case 16:
            switch (type.component_count) {
            case 1: return VK_FORMAT_R16_SFLOAT;
            case 2: return VK_FORMAT_R16G16_SFLOAT;
            case 3: return VK_FORMAT_R16G16B16_SFLOAT;
            case 4: return VK_FORMAT_R16G16B16A16_SFLOAT;
            }
        case 32:
            switch (type.component_count) {
            case 1: return VK_FORMAT_R32_SFLOAT;
            case 2: return VK_FORMAT_R32G32_SFLOAT;
            case 3: return VK_FORMAT_R32G32B32_SFLOAT;
            case 4: return VK_FORMAT_R32G32B32A32_SFLOAT;
            }
        }
    }
    return VK_FORMAT_UNDEFINED;
}

std::optional<InputInterfaceAttribute> FindVariable(const InputInterfaceAttributeList& list, 
                                                    const std::string& name) {
    std::optional<InputInterfaceAttribute> optional;

    return optional;
}

// TODO: can't handle - f16vec2, float16_t case
//
// In input attribute declared as VK_FORMAT_R16G16B16A16, in vertex (vec4)
// so, result always 32
InputInterfaceAttributeList GetInputInterfaceVariables(const ShaderModule& shader, const std::string& name)
{
    const auto& entries = shader.reflections.entry_points;
    auto it = std::find_if(entries.begin(), entries.end(), 
                           [&name](const auto& pts) { return name == pts.name; });
    ASSERT(it != entries.end());

    InputInterfaceAttributeList list;
    const auto& EntryPoint(*it);
    for (auto id : EntryPoint.interfaces_ids) {
        const auto& var = shader.reflections.variables.at(id);
        if (var.storage_class != SpvStorageClassInput) 
            continue;
        if (var.builtin.has_value()) 
            continue;
        const auto& primitive = shader.reflections.primitive_types.at(var.type_id);
        VkFormat format = GetPrimitiveFormat(primitive);
        uint32_t stride = GetPrimitiveStride(primitive);
        InputInterfaceAttribute att{format, var.location.value(), stride, var.name.value()};
        list.emplace_back(std::move(att));
    }

    std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.location < b.location; });

    return std::move(list);
}

const spirv::EntryPoint& GetEntryPoint(const ShaderModule& shader, const std::string& name) {
    const auto& entries = shader.reflections.entry_points;
    auto it = std::find_if(entries.begin(), entries.end(), [&name](const auto& pts) { return name == pts.name; });
    ASSERT(it != entries.end());
    const auto& EntryPoint(*it);
    return EntryPoint;
}

VariableRefList GetInterfaceVariableReferences(const ShaderModule& shader, const std::string& name) {
    auto& entry_point = GetEntryPoint(shader, name);
    std::vector<VariableRef> references;
    references.reserve(entry_point.interfaces_ids.size());
    for (auto id : entry_point.interfaces_ids)
        references.push_back(std::ref(shader.reflections.variables.at(id)));
    return std::move(references);
}

VkPipelineShaderStageCreateInfo GetPipelineShaderStageCreateInfo(const ShaderModule& shader, const std::string& name) {
    auto& entriy_point = GetEntryPoint(shader, name);
    
    VkPipelineShaderStageCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    info.stage = GetShaderStageBit(entriy_point.execution_model);
    info.module = shader.module;
    info.pName = entriy_point.name.c_str();

    return std::move(info);
}

std::vector<VkPushConstantRange> GetPushConstantRange(const ShaderModules shaders) {
    std::vector<VkPushConstantRange> ranges;
    for (auto& shader : shaders) {
        auto& reflection = shader.reflections;
        auto& entry_point = GetEntryPoint(shader, "main");
        for (auto& v : reflection.variables) {
            auto& var = v.second;
            if (var.storage_class != SpvStorageClassPushConstant)
                continue;
            auto& st = reflection.struct_types.at(var.type_id);
            auto offset = st.members.front().offset;
            auto size = GetStride(reflection, var.type_id);

            VkPushConstantRange range = {};
            range.stageFlags = GetShaderStageBit(entry_point.execution_model);
            range.size = size;
            range.offset = offset;
            ranges.push_back(range);
        }
    }

    // overlapping region requires both stage's flag in vkCmdPushConstants
    // so prohibit it
#if _DEBUG
    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    for (auto r : ranges)
        intervals.push_back({r.offset, r.size});
    std::sort(intervals.begin(), intervals.end());
    for (size_t i = 1; i < intervals.size(); i++) {
        uint32_t range_end = intervals[i-1].first + intervals[i-1].second;
        uint32_t next_range_start = intervals[i].first;
        ASSERT(range_end <= next_range_start);
    }
#endif
    return std::move(ranges);
}

