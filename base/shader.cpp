#include "shader.h"

#include <map>
#include <volk.h>
#include <variant>

#include "macro.h"

#ifndef POD_GET_MACRO
#define POD_GET_MACRO(Type, Name) \
    Type Name(size_t index) const \
    { \
        Type t; \
        Get(&t, index); \
        return t; \
    }
#endif

using word_t = uint32_t; // 32 bit

namespace internal {

struct StreamReader
{
    StreamReader(const void* data, size_t size) :
        data_(reinterpret_cast<const char*>(data)), last_(data_ + size)
    {
    }

    bool Eof() const
    {
        return data_ >= last_;
    }

    template <typename T>
    void Read(T* t)
    {
        // "underflow source buffer"
        ASSERT(data_ + sizeof(T) < last_);

        *t = *reinterpret_cast<const T*>(data_);
        data_ += sizeof(T);
    }

    // TODO: A bit confusing sizeof(T)*index statement
    template <typename T>
    void Get(T* t, size_t index) const
    {
        // "underflow source buffer"
        ASSERT(data_ + sizeof(T)*index < last_);

        *t = *(reinterpret_cast<const T*>(data_) + index);
    }

    size_t LiteralString(size_t index, size_t word_count) const
    {
        const char* begin = data_ + sizeof(word_t)*index;
        const char* end = data_ + sizeof(word_t)*(index + word_count);

        const char* it = begin;
        while (it < end) {
            if (*it++ == '\0') 
                break;
        }
        // final null char and tailing nulls - "main0000"
        size_t char_length = it - begin;
        size_t word_length = (char_length + sizeof(word_t) - 1) / sizeof(word_t);
        return word_length;
    }

    size_t LiteralString(size_t index, size_t word_count, std::string& str) const
    {
        size_t strlen_in_word = LiteralString(index, word_count);
        const char* begin = data_ + sizeof(word_t)*index;
        const char* end = data_ + sizeof(word_t)*(index + strlen_in_word);
        str = std::string(begin, end);
        return strlen_in_word;
    }

    void Skip(size_t index)
    {
        data_ += index;
    }

    const char* data_;
    const char* last_;

    POD_GET_MACRO(uint8_t, uint8)
    POD_GET_MACRO(uint16_t, uint16)
    POD_GET_MACRO(uint32_t, uint32)
};

VkShaderStageFlagBits GetShaderStage(SpvExecutionModel model)
{
    switch (model)
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
    ASSERT(FALSE);
}

struct VulkanFileHeader { 
    uint32_t VulkanMagicNumber;
    uint32_t VersionNumber;
    uint32_t GeneratorMasicNumber;
    uint32_t Bound;
    uint32_t Reserved;
};

struct Variable_
{
    uint32_t result_type_id;
    uint32_t result_id;
    SpvStorageClass storage_class;
};

struct Decoration_
{
    SpvDecoration decoration;
    uint32_t operand;
};

struct PointerType_
{
    uint32_t result_id;
    SpvStorageClass storage_class;
    uint32_t type_id;
};

struct IntegerType
{
    word_t result_id;
    word_t width;
    word_t signedness; // 0 unsigned, 1 signed
};

struct FloatType
{
    word_t result_id;
    word_t width;
};

struct VectorType
{
    uint32_t result_id;
    uint32_t component_type_id;
    uint32_t component_count;
};

using PrimitiveType_ = std::variant<IntegerType, FloatType, VectorType>;

struct IntermediateType {
    std::string entry_name;
    VkShaderStageFlagBits stage;
    std::unordered_map<uint32_t, std::vector<Decoration_>> decorations;
    std::unordered_map<uint32_t, Variable_> variables;
    std::unordered_map<uint32_t, PointerType_> pointer_types;
    std::map<uint32_t, PrimitiveType_> primitive_types; // process order dependent
    std::vector<word_t> interfaces_variables;
    std::unordered_map<uint32_t, std::string> names;
};

void ParseInstruction(word_t opcode, size_t word_count, const StreamReader& reader,
                      word_t id_bound, IntermediateType* intermediate) {
    switch (opcode)
    {
    case SpvOpEntryPoint: 
    {
        ASSERT(word_count >= 2);
        // OpEntryPoint Vertex %4 "main" %9 %11 %15 %17 %25 %28
        intermediate->stage = GetShaderStage(SpvExecutionModel(reader.uint32(1)));
        // reader.uint32(2); skip entry point id
        std::string str;
        size_t strlen_in_word = reader.LiteralString(3, word_count, str);
        size_t it = 3 + strlen_in_word;
        while (it < word_count)
            intermediate->interfaces_variables.push_back(reader.uint32(it++));
        intermediate->entry_name = str;
    } break;
    case SpvOpName:
    {
        ASSERT(word_count >= 3);
        uint32_t target_id = reader.uint32(1);
        std::string str;
        reader.LiteralString(2, word_count, str);
        intermediate->names.emplace(target_id, std::move(str));
    } break;
    case SpvOpDecorate:
    {
        ASSERT(word_count >= 3);
        uint32_t target_id = reader.uint32(1);
        ASSERT(target_id < id_bound);

        SpvDecoration decoration = SpvDecoration(reader.uint32(2));
        switch (decoration)
        {
        case SpvDecorationBuiltIn:
        case SpvDecorationBinding:
        case SpvDecorationLocation:
        case SpvDecorationDescriptorSet:
        {
            ASSERT(word_count == 4);
            intermediate->decorations[target_id].push_back({ decoration, reader.uint32(3) });
        } break;
        }
    } break;
    case SpvOpVariable:
    {
        ASSERT(word_count >= 4);
        uint32_t rid = reader.uint32(2);
        ASSERT(rid < id_bound);
        uint32_t rtid = reader.uint32(1);
        intermediate->variables.emplace(rid, Variable_ { rtid, rid, SpvStorageClass(reader.uint32(3)) });
    } break;
    case SpvOpTypePointer:
    {
        ASSERT(word_count == 4);
        // %10 = OpTypePointer Input %7
        uint32_t rid = reader.uint32(1);
        uint32_t tid = reader.uint32(3);
        intermediate->pointer_types.emplace(rid, PointerType_ { rid, SpvStorageClass(reader.uint32(2)), tid });

    } break;
    case SpvOpTypeInt:
    {
        ASSERT(word_count == 4);
        // %6 = OpTypeInt 32 0
        word_t rid = reader.uint32(1);
        word_t width = reader.uint32(2);
        word_t signedness = reader.uint32(3);
        intermediate->primitive_types.emplace(rid, IntegerType { rid, width, signedness });
    } break;
    case SpvOpTypeFloat:
    {
        ASSERT(word_count == 3);
        // %6 = OpTypeFloat 32
        word_t rid = reader.uint32(1);
        word_t width = reader.uint32(2);
        intermediate->primitive_types.emplace(rid, FloatType { rid, width });
    } break;
    case SpvOpTypeVector:
    {
        ASSERT(word_count == 4);
        // %7 = OpTypeVector %6 3
        uint32_t rid = reader.uint32(1);
        uint32_t ctid = reader.uint32(2);
        uint32_t cc = reader.uint32(3);
        intermediate->primitive_types.emplace(rid, VectorType { rid, ctid, cc });

    } break;
    }
}

void ParseSpirv(const void* data, size_t size, IntermediateType* intermediate) {
    ASSERT(intermediate);

    StreamReader reader(data, size);

    VulkanFileHeader header = {};
    reader.Read(&header);

    ASSERT(header.VulkanMagicNumber == SpvMagicNumber);

    uint32_t id_bound = header.Bound;

    while (!reader.Eof()) {
        uint16_t opcode = reader.uint16(0);
        uint16_t word_count = reader.uint16(1);
        ParseInstruction(opcode, word_count, reader, id_bound, intermediate);
        reader.Skip(sizeof(word_t)*word_count);
    }
}

} // namespace internal

class PrimitiveParserHelper {
public:

    PrimitiveParserHelper(ModuleType* module) : module_(module) {
    }

    void operator()(const internal::FloatType& type) {
        PrimitiveType p = {
            SpvOpTypeFloat,
            1,
            type.width,
        };
        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    void operator()(const internal::IntegerType& type) {
        PrimitiveType p = {
            SpvOpTypeInt,
            1,
            type.width,
            type.signedness
        };

        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    void operator()(const internal::VectorType& type) {
        PrimitiveType p = {};

        const auto& primitives = module_->primitive_types;
        ASSERT(primitives.count(type.component_type_id));
        const auto& component_type = primitives.find(type.component_type_id)->second;

        p.primitive_type = component_type.primitive_type;
        p.width = component_type.width;
        p.component_count = type.component_count;

        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    ModuleType* module_;
};

void VariableParser(const internal::Variable_& var, const internal::IntermediateType& intermediate,
                    ModuleType* module) {
    Variable v = {};
    uint32_t rid = var.result_id;
    v.type_id = var.result_type_id;
    v.storage_class = var.storage_class;

    auto pointer_it = intermediate.pointer_types.find(var.result_type_id);
    if (pointer_it != intermediate.pointer_types.end())
        v.type_id = pointer_it->second.type_id;

    auto deco_it = intermediate.decorations.find(rid);
    if (deco_it != intermediate.decorations.end()) {
        for (auto& deco : deco_it->second) {
            switch (deco.decoration)
            {
            case SpvDecorationBinding:
                v.binding = deco.operand; 
                break;
            case SpvDecorationDescriptorSet:
                v.descriptor_set = deco.operand; 
                break;
            case SpvDecorationLocation:
                v.location = deco.operand;
                break;
            case SpvDecorationBuiltIn:
                v.builtin = SpvBuiltIn(deco.operand);
                break;
            }
        }
    }

    auto name_it = intermediate.names.find(rid);
    if (name_it != intermediate.names.end())
        v.name = name_it->second;

    module->variables.emplace(rid, std::move(v));
    module->stroage_indices[SpvStorageClass(v.storage_class)].push_back(rid);
}

ModuleType ReflectShader(const void* data, size_t size) {
    internal::IntermediateType intermediate = {};
    internal::ParseSpirv(data, size, &intermediate);

    ModuleType module;
    PrimitiveParserHelper helper(&module);
    for (const auto& type : intermediate.primitive_types)
        std::visit(helper, type.second);

    for (const auto& var : intermediate.variables) 
        VariableParser(var.second, intermediate, &module);
    return module;
}

