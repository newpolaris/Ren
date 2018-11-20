#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec4 color;

void main()
{
    gl_Position = vec4(position/3 + vec3(0.0, 0.0, 0.5), 1.0);
    color = vec4(normal * 0.5 + vec3(0.5), 1.0);
}
