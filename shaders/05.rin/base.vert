#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_GOOGLE_include_directive: require 

struct Vertex {
    float x, y, z;
    float nx, ny, nz;
    float tu, tv;
};

#if FVF
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;
#else
layout(binding = 0) buffer block {
    Vertex vertices[];
};
#endif

layout(location = 0) out vec3 color;

struct MeshDraw
{
	vec2 offset;
	vec2 scale;
};

layout(push_constant) uniform param_block {
    MeshDraw mesh_draw;
};

void main() {
    vec3 scale = vec3(mesh_draw.scale, 0.5);
    vec3 offset = vec3(mesh_draw.offset, 0.0);
#if !FVF
    vec3 position = vec3(vertices[gl_VertexIndex].x, vertices[gl_VertexIndex].y, vertices[gl_VertexIndex].z); 
    vec3 normal = vec3(vertices[gl_VertexIndex].nx, vertices[gl_VertexIndex].ny, vertices[gl_VertexIndex].nz); 
    vec2 texcoords = vec2(vertices[gl_VertexIndex].tu, vertices[gl_VertexIndex].tv);
#endif
    color = normal;
    gl_Position = vec4((position * scale + offset) * vec3(2.0, 2.0, 0.0) + vec3(-1.0, -1.0, 0.5), 1.0);
}