#include "program.h"

#include <algorithm>
#include "spirv-reflect.h"
#include "macro.h"
#include "resources.h"

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

InputInterfaceAttributeList GetInputInterfaceVariables(const ShaderModule& shader, const std::string& name);
VariableRefList GetInterfaceVariableReferences(const ShaderModule& shader, const std::string& name);
std::optional<InputInterfaceAttribute> FindVariable(const InputInterfaceAttributeList& list, const std::string& name);

VkPipelineShaderStageCreateInfo GetPipelineShaderStageCreateInfo(const ShaderModule& shader, const std::string& name);
std::vector<VkPushConstantRange> GetPushConstantRange(const ShaderModules shaders);


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

PushDescriptorSets::PushDescriptorSets(const Buffer & buffer) {
    buffer_.buffer = buffer.buffer;
    buffer_.offset = 0;
    buffer_.range = buffer.size;
}

VkPipelineLayout CreatePipelineLayout(VkDevice device, ShaderModules shaders) {
    std::vector<VkShaderStageFlags> stages(32);

    for (auto& shader : shaders) 
        for (auto& var : shader.reflections.variables) {
            VkShaderStageFlags flags = 0;
            if (shader.reflections.entry_points.front().execution_model == SpvExecutionModelVertex)
                flags = VK_SHADER_STAGE_VERTEX_BIT;
            if (var.second.storage_class == SpvStorageClassUniform) 
                stages[var.second.binding.value()] |= flags;
        }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (size_t i = 0; i < stages.size(); i++) {
        if (stages[i] == 0)
            continue;
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = stages[i];
        bindings.push_back(binding);
    }


    VkDescriptorSetLayoutCreateInfo set_create_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    set_create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    set_create_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_create_info.pBindings = bindings.data();

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDescriptorSetLayout(device, &set_create_info, nullptr, &set_layout));

    std::vector<VkPushConstantRange> ranges = GetPushConstantRange(shaders);

    VkPipelineLayoutCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    info.setLayoutCount = 1;
    info.pSetLayouts = &set_layout;
    info.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
    info.pPushConstantRanges = ranges.data();

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_ASSERT(vkCreatePipelineLayout(device, &info, nullptr, &layout));

    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);

    return layout;
}

VkPipeline CreatePipeline(VkDevice device, VkPipelineLayout layout, VkRenderPass pass, ShaderModules shaders) {
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    for (auto& shader : shaders)
        stages.push_back(GetPipelineShaderStageCreateInfo(shader, "main"));

    VkPipelineVertexInputStateCreateInfo vertex_input_info { 
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO 
    };

    VkPipelineInputAssemblyStateCreateInfo input_info { 
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO 
    };
    input_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_info { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_info.viewportCount = 1;
    viewport_info.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster_info = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster_info.lineWidth = 1.0;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisample_info = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendstate = {};
    blendstate.blendEnable = VK_FALSE;
    blendstate.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend_info = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = &blendstate;

    VkDynamicState dynamic_state[] { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_info = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_info.dynamicStateCount = ARRAY_SIZE(dynamic_state);
    dynamic_info.pDynamicStates = dynamic_state;

    VkGraphicsPipelineCreateInfo info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    info.stageCount = static_cast<uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_input_info;
    info.pInputAssemblyState = &input_info;
    info.pViewportState = &viewport_info;
    info.pRasterizationState = &raster_info;
    info.pMultisampleState = &multisample_info;
    info.pColorBlendState = &blend_info;
    info.pDynamicState = &dynamic_info;

    info.layout = layout;
    info.renderPass = pass;
    info.subpass = 0;

    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &info, nullptr, &pipeline));

    return pipeline;
}

VkDescriptorUpdateTemplate CreateDescriptorUpdateTemplate(VkDevice device, VkPipelineLayout layout, 
                                                          ShaderModules shaders) {
    std::vector<VkDescriptorType> bindings(32, VK_DESCRIPTOR_TYPE_MAX_ENUM);

    for (auto& shader : shaders) 
        for (auto& var : shader.reflections.variables)
            if (var.second.storage_class == SpvStorageClassUniform)
                bindings[var.second.binding.value()] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    std::vector<VkDescriptorUpdateTemplateEntry> entries;
    for (size_t i = 0; i < bindings.size(); i++) {
        if (bindings[i] == VK_DESCRIPTOR_TYPE_MAX_ENUM) 
            continue;
        VkDescriptorUpdateTemplateEntry entry = {};
        entry.dstBinding = static_cast<uint32_t>(i);
        entry.descriptorCount = 1;
        entry.descriptorType = bindings[i];
        entry.offset = entries.size() * sizeof(PushDescriptorSets);
        entry.stride = sizeof(PushDescriptorSets);
        entries.push_back(entry);
    }

    VkDescriptorUpdateTemplateCreateInfo info = { VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO };
    info.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
    info.pDescriptorUpdateEntries = entries.data();
    info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    info.descriptorSetLayout = VK_NULL_HANDLE;
    info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    info.pipelineLayout = layout;
    info.set = 0;

    VkDescriptorUpdateTemplate descriptorUpdateTemplate = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDescriptorUpdateTemplate(device, &info, nullptr, &descriptorUpdateTemplate));
    return descriptorUpdateTemplate;
}
