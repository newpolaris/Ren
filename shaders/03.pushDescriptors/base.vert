#version 450
#extension GL_ARB_separate_shader_objects : enable

struct Vertex
{
    float x, y, z;
    float nx, ny, nz;
    float tu, uv;
};

layout(binding = 0) readonly buffer Vertices
{
    Vertex vertices[];
};

layout(location = 0) out vec4 color;

void main()
{
    Vertex v = vertices[gl_VertexIndex];
    vec3 position = vec3(v.x, v.y, v.z);
    vec3 normal = vec3(v.nx, v.ny, v.nz);

    gl_Position = vec4(position/3 + vec3(0.0, 0.0, 0.5), 1.0);
    color = vec4(normal * 0.5 + vec3(0.5), 1.0);
}
