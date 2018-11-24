#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in uvec4 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec4 color;

void main()
{
    gl_Position = vec4(position*vec3(0.4, 0.4, 0.1) + vec3(0.0, -0.7, 0.5), 1.0);
    color = vec4((vec3(normal.xyz) / 127.0 - 1.0) * 0.5 + vec3(0.5), 1.0);
}
