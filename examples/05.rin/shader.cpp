#include "shader.h"

#include <vulkan/spirv.h>
#include <unordered_map>
#include <volk.h>

#include "common.h"

#ifndef POD_GET_MACRO
#define POD_GET_MACRO(Type, Name) \
    Type Name(size_t index) \
    { \
        Type t; \
        get(&t, index); \
        return t; \
    }
#endif

struct StreamReader
{
    StreamReader(const void* data, size_t size) :
        data_(reinterpret_cast<const char*>(data)), last_(data_ + size)
    {
    }

    bool eof() const
    {
        return data_ >= last_;
    }

    template <typename T>
    void read(T* t)
    {
        // "underflow source buffer"
        ASSERT(data_ + sizeof(T) < last_);

        *t = *reinterpret_cast<const T*>(data_);
        data_ += sizeof(T);
    }

    template <typename T>
    void get(T* t, size_t index) const
    {
        // "underflow source buffer"
        ASSERT(data_ + sizeof(T)*index < last_);

        *t = *(reinterpret_cast<const T*>(data_) + index);
    }

    void skip(size_t index)
    {
        data_ += index;
    }

    const char* data_;
    const char* last_;

    POD_GET_MACRO(uint8_t, uint8)
    POD_GET_MACRO(uint16_t, uint16)
    POD_GET_MACRO(uint32_t, uint32)
};

VkShaderStageFlagBits getShaderStage(SpvExecutionModel executionModel)
{
    switch (executionModel)
    {
        case SpvExecutionModelVertex:
            return VK_SHADER_STAGE_VERTEX_BIT;
        case SpvExecutionModelFragment:
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        case SpvExecutionModelTaskNV:
            return VK_SHADER_STAGE_TASK_BIT_NV;
        case SpvExecutionModelMeshNV:
            return VK_SHADER_STAGE_MESH_BIT_NV;
    }
    throw std::runtime_error("fail to get shader stage");
}

struct VulkanFileHeader { 
    uint32_t VulkanMagicNumber;
    uint32_t VersionNumber;
    uint32_t GeneratorMasicNumber;
    uint32_t Bound;
    uint32_t Reserved;
};

enum SpvTypes : uint32_t
{
    SpvUnknown = 0,
    SpvVariable,
};

namespace intermediate
{
    struct Variable
    {
        uint32_t type_id;
        SpvStorageClass_ storage_class;
    };

    struct Decoration
    {
        SpvDecoration decoration;
        uint32_t operand;
    };
};

struct Variable
{
    uint32_t type_id;
    uint32_t storage_class;
    uint32_t binding;
    uint32_t location;
    uint32_t descriptor_set;
};

void ParseShader(const void* data, size_t size)
{
    VkShaderStageFlagBits stage_;
    uint32_t storage_buffer_mask_;

    StreamReader reader(data, size);

    VulkanFileHeader header = {};
    reader.read(&header);

    ASSERT(header.VulkanMagicNumber == SpvMagicNumber);

    uint32_t idBound = header.Bound;

    std::unordered_map<uint32_t, std::vector<intermediate::Decoration>> decorations;
	std::unordered_map<uint32_t, intermediate::Variable> variables;

    while (!reader.eof())
    {
        uint16_t opcode = reader.uint16(0);
        uint16_t wordCount = reader.uint16(1);

        switch (opcode)
        {
        case SpvOpEntryPoint: 
        {
			ASSERT(wordCount >= 2);
            stage_ = getShaderStage(SpvExecutionModel(reader.uint32(1)));
        } break;
        case SpvOpDecorate:
        {
            ASSERT(wordCount >= 3);
            uint32_t targetid = reader.uint32(1);
            ASSERT(targetid < idBound);

            SpvDecoration decoration = SpvDecoration(reader.uint32(2));
            switch (decoration)
            {
            case SpvDecorationBinding:
            case SpvDecorationLocation:
            case SpvDecorationDescriptorSet:
            {
                ASSERT(wordCount == 4);
                decorations[targetid].push_back({ decoration, reader.uint32(3) });
            } break;
            }
        } break;
        case SpvOpVariable:
        {
            ASSERT(wordCount >= 4);
            uint32_t resultid = reader.uint32(2);
            ASSERT(resultid < idBound);
            variables.emplace(resultid, intermediate::Variable{ reader.uint32(1), SpvStorageClass_(reader.uint32(3)) });
        } break;
        }

        reader.skip(sizeof(uint32_t)*wordCount);
    }

    std::vector<Variable> result_variables;
    for (auto& var : variables)
    {
        Variable v = {};

        v.type_id = var.second.type_id;
        v.storage_class = var.second.storage_class;

        auto it = decorations.find(var.first);
        if (it == decorations.end())
            continue;

        for (auto& deco : it->second)
        {
            switch (deco.decoration)
            {
            case SpvDecorationBinding:
                v.binding = deco.operand; 
                break;
            case SpvDecorationDescriptorSet:
                v.descriptor_set = deco.operand; 
                break;
            }
        }
        result_variables.push_back(v);
    }

    for (auto& var : result_variables)
    {
        ASSERT(var.descriptor_set == 0);
        ASSERT(var.binding < 32);

        storage_buffer_mask_ |= 1 << var.binding;
    }
}

