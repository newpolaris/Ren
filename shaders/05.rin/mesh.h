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

struct MeshDrawCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct MeshDraw
{
    vec3 position;
    float scale;
    vec4 orientation;
    vec3 center;
    float radius;
    uint vertex_offset;
    uint meshlet_offset;
    uint meshlet_count;
    uint pad[1];
};

struct Meshlet
{
    vec3 center;
    float radius;
    uint index_offset;
    i8vec4 cone;
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct MeshletData
{
    uint vertices[64];
    uint8_t indices[126*3];
};

struct CullingData
{
    vec4 frustum[6];
    uint draw_count;
    uint pad[3];
};
