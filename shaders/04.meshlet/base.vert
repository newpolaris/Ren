#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_GOOGLE_include_directive : require

#include "mesh.h"

#define DEBUG 0

layout(binding = 0) readonly buffer Vertices
{
    Vertex vertices[];
};

layout(location = 0) out vec4 color;

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

void main()
{
    vec3 position = vec3(vertices[gl_VertexIndex].x, vertices[gl_VertexIndex].y, vertices[gl_VertexIndex].z);
    vec3 normal = vec3(int(vertices[gl_VertexIndex].nx), int(vertices[gl_VertexIndex].ny), int(vertices[gl_VertexIndex].nz)) / 127.0 - 1.0;

    gl_Position = vec4(position*vec3(1.0, 1.0, 0.5) + vec3(0.0, 0.0, 0.5), 1.0);
    color = vec4(normal * 0.5 + vec3(0.5), 1.0);

#if DEBUG
    uint mi = gl_InstanceIndex;
	uint mhash = hash(mi);
	color = vec4(vec3(float(mhash & 255), float((mhash >> 8) & 255), float((mhash >> 16) & 255)) / 255.0, 1.0);
#endif

}
