#version 450
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require

#extension GL_GOOGLE_include_directive: require 

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

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
    color = normal;
    vec3 scale = vec3(mesh_draw.scale, 0.5);
    vec3 offset = vec3(mesh_draw.offset, 0.0);
    gl_Position = vec4((position * scale + offset) * vec3(2.0, 2.0, 0.0) + vec3(-1.0, -1.0, 0.5), 1.0);
}