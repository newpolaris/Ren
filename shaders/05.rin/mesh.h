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
    uint indirect_command[5];
};
