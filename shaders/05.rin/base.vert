#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require 

#define FVF 0

struct Vertex {
    vec3 position; 
    u8vec3 normal;
    f16vec2 texcoords;
};

#if FVF
layout(location = 0) in vec3 position;
layout(location = 1) in uvec3 normal;
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
    // To handle "Capability Int8 is not allowed by Vulkan 1.1 specification (or requires extension) OpCapability Int8"
    // Not allowed assignment / direct cast to float (allowed via int) (maybe? with/without both raise validation error)
    vec3 position = vec3(vertices[gl_VertexIndex].position); 
    vec3 normal = vec3(uvec3(vertices[gl_VertexIndex].normal)); 
    vec2 texcoords = vec2(vertices[gl_VertexIndex].texcoords);
#endif
    color = vec3(normal) / 127.0 - 1.0;
    gl_Position = vec4((position * scale + offset) * vec3(2.0, 2.0, 0.0) + vec3(-1.0, -1.0, 0.5), 1.0);
}