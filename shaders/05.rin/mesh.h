const uint max_meshlet_count = 1000;

struct Vertex {
    vec3 position;
    u8vec3 normal;
    f16vec2 texcoords;
};

struct GraphicsData
{
    mat4x4 project;
};

struct MeshletCall {
    uint mesh_id;
    uint index;
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


struct MeshDraw
{
    vec3 position;
    float scale;
    vec4 orientation;
    vec3 center;
    float radius;
    uint index_count;
    uint meshlet_offset;
    uint meshlet_count;
    uint pad[1];
};

struct MeshletDraw
{
    vec3 center;
    float radius;
    uint index_offset;
    uint index_count;
    i8vec4 cone;
    uint pad[1];
};

struct CullingData
{
    vec4 frustum[6];
    uint draw_count;
    uint pad[3];
};
