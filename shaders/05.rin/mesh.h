struct Vertex {
    vec3 position;
    u8vec3 normal;
    f16vec2 texcoords;
};

struct PushConstant
{
    mat4x4 project;
};

struct MeshDraw
{
    vec3 position;
    float scale;
    vec4 orientation;
    uint index_count;
};

// Aka. VkDrawIndexedIndirectCommand
struct MeshDrawCommand 
{
    uint index_count;
    uint instance_count;
    uint first_index;
    uint vertex_offset;
    uint first_instance;
};
