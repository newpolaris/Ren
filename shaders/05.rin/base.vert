#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require 

struct Vertex {
    vec3 position; 
    u8vec3 normal;
    f16vec2 texcoords;
};

layout(binding = 0) buffer block {
    Vertex vertices[];
};

layout(location = 0) out vec3 color;

struct MeshDraw
{
    mat4x4 project;
	vec3 position;
    float scale;
    vec4 orientation;
};

layout(push_constant) uniform param_block {
    MeshDraw mesh_draw;
};

vec3 rotate_position(vec4 quat, vec3 v) {
    return v + 2.0 * cross(quat.xyz, cross(quat.xyz, v) + quat.w * v);
 }

void main() {
    // To handle "Capability Int8 is not allowed by Vulkan 1.1 specification (or requires extension) OpCapability Int8"
    // Not allowed assignment / direct cast to float (allowed via int) (maybe? with/without both raise validation error)
    vec3 position = vec3(vertices[gl_VertexIndex].position); 
    vec3 normal = vec3(uvec3(vertices[gl_VertexIndex].normal)); 
    vec2 texcoords = vec2(vertices[gl_VertexIndex].texcoords);

    color = vec3(normal) / 127.0 - 1.0;
    gl_Position = mesh_draw.project * vec4(rotate_position(mesh_draw.orientation, position) * mesh_draw.scale + mesh_draw.position, 1.0);
}