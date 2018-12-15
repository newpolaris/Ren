#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_GOOGLE_include_directive: require 

#include "mesh.h"

layout(push_constant) uniform param_block {
    GraphicsData constant;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(binding = 1) readonly buffer Indices {
    uint indices[];
};

layout(binding = 2) readonly buffer MeshDraws {
    MeshDraw mesh_draws[];
};

layout(location = 0) out vec3 color;

vec3 rotate_position(vec4 quat, vec3 v) {
    return v + 2.0 * cross(quat.xyz, cross(quat.xyz, v) + quat.w * v);
 }

void main() {
    MeshDraw draws = mesh_draws[gl_InstanceIndex];

    uint index = gl_VertexIndex;
    uint vertex_id = indices[index];

    // To handle "Capability Int8 is not allowed by Vulkan 1.1 specification (or requires extension) OpCapability Int8"
    // Not allowed assignment / direct cast to float (allowed via int) (maybe? with/without both raise validation error)
    vec3 position = vec3(vertices[vertex_id].position); 
    vec3 normal = vec3(uvec3(vertices[vertex_id].normal)); 
    vec2 texcoords = vec2(vertices[vertex_id].texcoords);

    color = vec3(normal) / 127.0 - 1.0;
    gl_Position = constant.project * vec4(rotate_position(draws.orientation, position) * draws.scale + draws.position, 1.0);
}