// references
//
// [vulkan-cpp-library]
// [spirv_reflect]

#include "spirv-reflect.h"

#include <map>
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

namespace spirv {
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
        str = std::string(begin);
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
    SpvDecoration decorator;
    uint32_t operand;
};

struct MemberDecoration_
{
    SpvDecoration decorator;
    uint32_t member;
    std::optional<uint32_t> operand;
};

struct Constant_
{
    uint32_t result_type_id;
    uint32_t result_id;
    uint32_t value;
};

struct PointerType_
{
    uint32_t result_id;
    SpvStorageClass storage_class;
    uint32_t type_id;
};

struct IntegerType_
{
    word_t result_id;
    word_t width;
    word_t signedness; // 0 unsigned, 1 signed
};

struct FloatType_
{
    word_t result_id;
    word_t width;
};

struct VectorType_
{
    uint32_t result_id;
    uint32_t component_type_id;
    uint32_t component_count;
};

struct StructType_
{
    word_t result_id;
    std::vector<word_t> member_ids;
};

struct ArrayType_
{
    uint32_t result_id;
    uint32_t element_type_id;
    uint32_t length_id;
};

struct MatrixType_
{
    word_t result_id;
    uint32_t component_type_id;
    uint32_t component_count;
};

using Name_ = std::string;
using MemberName_ = std::unordered_map<word_t, std::string>;
using PrimitiveType_ = std::variant<IntegerType_, FloatType_, VectorType_, MatrixType_>;

// borrowed from 'vulkan-cpp-library'
struct IntermediateType {
    std::vector<EntryPoint> entry_points;
    std::unordered_map<uint32_t, std::vector<Decoration_>> decorations;
    std::unordered_map<uint32_t, std::vector<MemberDecoration_>> member_decorations;
    std::unordered_map<uint32_t, Constant_> constants;
    std::unordered_map<uint32_t, Variable_> variables;
    std::unordered_map<uint32_t, PointerType_> pointer_types;
    std::map<uint32_t, PrimitiveType_> primitive_types; // process order dependent
    std::map<uint32_t, StructType_> struct_types; 
    std::map<uint32_t, ArrayType_> array_types;
    std::unordered_map<uint32_t, Name_> names;
    std::unordered_map<uint32_t, MemberName_> member_names;
};

void ParseInstruction(word_t opcode, size_t word_count, const StreamReader& reader,
                      word_t id_bound, IntermediateType* intermediate) {
    switch (opcode)
    {
    case SpvOpEntryPoint: 
    {
        ASSERT(word_count >= 2);
        // OpEntryPoint Vertex %4 "main" %9 %11 %15 %17 %25 %28
        std::string name;
        size_t strlen_in_word = reader.LiteralString(3, word_count, name);
        size_t it = 3 + strlen_in_word;
        std::vector<word_t> ids;
        while (it < word_count)
            ids.push_back(reader.uint32(it++));

        EntryPoint pt { 
            SpvExecutionModel(reader.uint32(1)),
            reader.uint32(2),
            name,
            ids
        };
        intermediate->entry_points.push_back(std::move(pt));
    } break;
    case SpvOpName:
    {
        ASSERT(word_count >= 3);
        uint32_t target_id = reader.uint32(1);
        std::string str;
        reader.LiteralString(2, word_count, str);
        intermediate->names.emplace(target_id, std::move(str));
    } break;
    case SpvOpMemberName:
    {
        ASSERT(word_count >= 4);
        uint32_t type_id = reader.uint32(1);
        uint32_t member = reader.uint32(2);
        std::string str;
        reader.LiteralString(3, word_count, str);
        intermediate->member_names[type_id].emplace(member, std::move(str));
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
    case SpvOpMemberDecorate:
    {
        ASSERT(word_count >= 3);
        uint32_t struct_type_id = reader.uint32(1);
        ASSERT(struct_type_id < id_bound);

        uint32_t member = reader.uint32(2);
        SpvDecoration decoration = SpvDecoration(reader.uint32(3));
        switch (decoration)
        {
        case SpvDecorationOffset:
        case SpvDecorationMatrixStride:
        case SpvDecorationBuiltIn:
        {
            ASSERT(word_count == 5);
            intermediate->member_decorations[struct_type_id].push_back(
                { decoration, member, reader.uint32(4) });
        } break;
        case SpvDecorationRowMajor:
        case SpvDecorationColMajor:
        {
            ASSERT(word_count == 4);
            intermediate->member_decorations[struct_type_id].push_back(
                { decoration, member });
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
    case SpvOpConstant:
    {
        ASSERT(word_count >= 4);
        // %27 = OpConstant %6 3
        uint32_t rid = reader.uint32(2);
        ASSERT(rid < id_bound);
        uint32_t rtid = reader.uint32(1);
        word_t value = reader.uint32(3); // TODO: at least one word, can be multiple words
        intermediate->constants.emplace(rid, Constant_ { rtid, rid, value });
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
        intermediate->primitive_types.emplace(rid, IntegerType_ { rid, width, signedness });
    } break;
    case SpvOpTypeFloat:
    {
        ASSERT(word_count == 3);
        // %6 = OpTypeFloat 32
        word_t rid = reader.uint32(1);
        word_t width = reader.uint32(2);
        intermediate->primitive_types.emplace(rid, FloatType_ { rid, width });
    } break;
    case SpvOpTypeVector:
    {
        ASSERT(word_count == 4);
        // %7 = OpTypeVector %6 3
        uint32_t rid = reader.uint32(1);
        uint32_t ctid = reader.uint32(2);
        uint32_t cc = reader.uint32(3);
        intermediate->primitive_types.emplace(rid, VectorType_ { rid, ctid, cc });
    } break;
    case SpvOpTypeMatrix:
    {
        ASSERT(word_count == 4);
        // %114 = OpTypeMatrix %7 4
        uint32_t rid = reader.uint32(1);
        uint32_t ctid = reader.uint32(2);
        uint32_t cc = reader.uint32(3);
        intermediate->primitive_types.emplace(rid, MatrixType_ { rid, ctid, cc });
    } break;
    case SpvOpTypeStruct:
    {
        ASSERT(word_count >= 2);
        // %44 = OpTypeStruct %43 %43 %43
        uint32_t rid = reader.uint32(1);
        std::vector<word_t> member_ids;
        for (uint32_t it = 2; it < word_count; it++)
            member_ids.push_back(reader.uint32(it));
        intermediate->struct_types.emplace(rid, StructType_ { rid, member_ids });
    } break;
    case SpvOpTypeArray:
    {
        ASSERT(word_count == 4);
        // %28 = OpTypeArray %6 %27
        uint32_t rid = reader.uint32(1);
        uint32_t etid = reader.uint32(2);
        uint32_t lid = reader.uint32(3); // spec: integer type const op with at least 1
        intermediate->array_types.emplace(rid, ArrayType_ { rid, etid, lid });
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

    void operator()(const internal::FloatType_& type) {
        PrimitiveType p = {
            SpvOpTypeFloat,
            {1, 1},
            type.width,
        };
        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    void operator()(const internal::IntegerType_& type) {
        PrimitiveType p = {
            SpvOpTypeInt,
            {1, 1},
            type.width,
            type.signedness
        };

        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    void operator()(const internal::VectorType_& type) {
        PrimitiveType p = {};

        const auto& primitives = module_->primitive_types;
        ASSERT(primitives.count(type.component_type_id));
        const auto& component_type = primitives.find(type.component_type_id)->second;

        p.primitive_type = component_type.primitive_type;
        p.width = component_type.width;
        p.component_count[0] = type.component_count;
        p.component_count[1] = 1;

        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    void operator()(const internal::MatrixType_& type) {
        PrimitiveType p = {};

        const auto& primitives = module_->primitive_types;
        ASSERT(primitives.count(type.component_type_id));
        const auto& vector_type = primitives.find(type.component_type_id)->second;

        p.primitive_type = vector_type.primitive_type;
        p.width = vector_type.width;
        p.component_count[0] = vector_type.component_count[0];
        p.component_count[1] = type.component_count;

        module_->primitive_types.emplace(type.result_id, std::move(p));
    }

    ModuleType* module_;
};

void DecorateStruct(const internal::MemberDecoration_& deco, MemberType* m) {
    switch (deco.decorator)
    {
    case SpvDecorationColMajor:
        m->col_major = 1;
        break;
    case SpvDecorationOffset:
        m->offset = deco.operand.value();
        break;
    case SpvDecorationMatrixStride:
        m->matrix_stride = deco.operand.value();
        break;
    case SpvDecorationBuiltIn:
        m->builtin = deco.operand.value();
        break;
    }
}

void ParseArray(const internal::ArrayType_& arr, const internal::IntermediateType& intermediate,
                 ModuleType* module) {
    uint32_t result_id = arr.result_id;
    uint32_t element_type_id = arr.element_type_id;
    uint32_t length = arr.length_id;
    ASSERT(length >= 1);
    module->array_types.emplace(result_id, ArrayType { element_type_id, length });
}

void ParseStruct(const internal::StructType_& var, const internal::IntermediateType& intermediate,
                 ModuleType* module) {
    uint32_t result_id = var.result_id;

    std::vector<MemberType> members;

    for (size_t i = 0; i < var.member_ids.size(); i++) {
        const auto& names = intermediate.member_names.at(result_id);

        MemberType member = {};
        member.type_id = var.member_ids[i];
        member.name = names.at(i);
        members.emplace_back(std::move(member));
    }

    auto deco_it = intermediate.member_decorations.find(result_id);
    if (deco_it != intermediate.member_decorations.end()) {
        for (auto& deco : deco_it->second)
            DecorateStruct(deco, &members[deco.member]);
    }
    std::optional<std::string> name;
    auto name_it = intermediate.names.find(result_id);
    if (name_it != intermediate.names.end())
        name = name_it->second;

    module->struct_types.emplace(result_id, StructType { name, members });
}

void DecorateVariable(const internal::Decoration_& deco, Variable* v) {
    switch (deco.decorator)
    {
    case SpvDecorationBinding:
        v->binding = deco.operand;
        break;
    case SpvDecorationDescriptorSet:
        v->descriptor_set = deco.operand;
        break;
    case SpvDecorationLocation:
        v->location = deco.operand;
        break;
    case SpvDecorationBuiltIn:
        v->builtin = SpvBuiltIn(deco.operand);
        break;
    }
}

void ParseConstant(const internal::Constant_& cons, const internal::IntermediateType& intermediate,
                   ModuleType* module) {
    Constant c = {};
    c.type_id = cons.result_type_id;
    c.value = cons.value;

    uint32_t rid = cons.result_id;
    module->constants.emplace(rid, std::move(c));
}

void ParseVariable(const internal::Variable_& var, const internal::IntermediateType& intermediate,
                   ModuleType* module) {
    Variable v = {};
    uint32_t rid = var.result_id;
    v.type_id = var.result_type_id;
    v.storage_class = var.storage_class;

    auto pointer_it = intermediate.pointer_types.find(var.result_type_id);
    if (pointer_it != intermediate.pointer_types.end())
        v.type_id = pointer_it->second.type_id;

    auto deco_it = intermediate.decorations.find(rid);
    if (deco_it != intermediate.decorations.end())
        for (const auto& deco : deco_it->second)
            DecorateVariable(deco, &v);

    auto name_it = intermediate.names.find(rid);
    if (name_it != intermediate.names.end())
        v.name = name_it->second;

    module->variables.emplace(rid, std::move(v));

    // fill cached indices
    module->stroage_indices[SpvStorageClass(v.storage_class)].push_back(rid);
}

SpirvReflections ReflectShader(const void* data, size_t size) {
    internal::IntermediateType intermediate = {};
    internal::ParseSpirv(data, size, &intermediate);

    ModuleType module;
    for (const auto& type : intermediate.constants)
        ParseConstant(type.second, intermediate, &module);

    PrimitiveParserHelper helper(&module);
    for (const auto& type : intermediate.primitive_types)
        std::visit(helper, type.second);

    for (const auto& type : intermediate.array_types)
        ParseArray(type.second, intermediate, &module);

    for (const auto& type : intermediate.struct_types)
        ParseStruct(type.second, intermediate, &module);

    for (const auto& var : intermediate.variables) 
        ParseVariable(var.second, intermediate, &module);

    module.entry_points = intermediate.entry_points;

    return module;
}

} // namespace spirv {
