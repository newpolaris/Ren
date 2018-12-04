#pragma once

#include <vulkan/spirv.h>
#include <cstdint>
#include <optional>
#include <unordered_map>

struct PrimitiveType
{
    SpvOp primitive_type;
    uint32_t component_count;
    std::optional<uint32_t> width;
    std::optional<uint32_t> signedness;
};

struct Variable
{
    uint32_t type_id;
    SpvStorageClass storage_class;
    std::optional<uint32_t> binding;
    std::optional<uint32_t> location;
    std::optional<uint32_t> descriptor_set;
    std::optional<SpvBuiltIn> builtin;
    std::optional<std::string> name;
};

struct ModuleType
{
    std::unordered_map<uint32_t, PrimitiveType> primitive_types;
    std::unordered_map<uint32_t, Variable> variables;
    std::unordered_map<SpvStorageClass, std::vector<uint32_t>> stroage_indices;
};

ModuleType ReflectShader(const void* data, size_t size);
