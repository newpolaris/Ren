#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

const vec3 vertices[] = 
{
	vec3(0.0, 0.5, 0.0),
	vec3(0.5, -0.5, 0.0),
	vec3(-0.5, -0.5, 0.0),
};

void main() {
    gl_Position = vec4(position, 1.0);
}
