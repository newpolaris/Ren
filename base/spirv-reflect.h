#pragma once

#include <vulkan/spirv.h>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace spirv {

struct PrimitiveType
{
    SpvOp primitive_type;
    uint32_t component_count[2];
    std::optional<uint32_t> width;
    std::optional<uint32_t> signedness; // 0 unsigned, 1 signed; TODO
};

struct MemberType
{
    uint32_t type_id;
    uint32_t offset;
    std::string name;
    std::optional<uint32_t> matrix_stride;
    std::optional<uint32_t> builtin;
    std::optional<uint32_t> col_major; // TODO
};

struct ArrayType
{
    uint32_t element_type_id;
    uint32_t length_id;
};

struct StructType
{
    std::optional<std::string> name;
    std::vector<MemberType> members;
};

struct MatrixType
{
    std::optional<std::string> name;
};

struct Constant
{
    uint32_t type_id;
    uint32_t value;
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

struct EntryPoint 
{
	SpvExecutionModel execution_model;
	uint32_t entry_point_id;
	std::string name;
	std::vector<uint32_t> interfaces_ids;
};

struct ModuleType
{
    std::vector<EntryPoint> entry_points;
    std::unordered_map<uint32_t, PrimitiveType> primitive_types;
    std::unordered_map<uint32_t, ArrayType> array_types;
    std::unordered_map<uint32_t, StructType> struct_types;
    std::unordered_map<uint32_t, MatrixType> matrix_types;
    std::unordered_map<uint32_t, Constant> constants;
    std::unordered_map<uint32_t, Variable> variables;
    std::unordered_map<SpvStorageClass, std::vector<uint32_t>> stroage_indices;
};

using SpirvReflections = ModuleType;

SpirvReflections ReflectShader(const void* data, size_t size);

} // namespace spirv