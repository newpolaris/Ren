struct Vertex {
    vec3 position;
    u8vec3 normal;
    f16vec2 texcoords;
};

struct GraphicsData
{
    mat4x4 project;
};

struct MeshDraw
{
    vec3 position;
    float scale;
    vec4 orientation;
    vec3 center;
    float radius;
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

struct CullingData
{
    vec4 frustum0, frustum1, frustum2, frustum3, frustum4, frustum5;
    uint draw_count;
    uint pad0, pad1, pad2;
};
