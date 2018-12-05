#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec3 color_;
layout(location = 1) out vec2 texcoord_;

layout(binding = 0) uniform UniformBufferObject {
    mat4 _model_;
    mat4 _view_;
    mat4 _proj_;
} ubo;

void main() {
    color_ = normal;
    texcoord_ = texcoord;
    gl_Position = vec4(position * vec3(1, 1, 0.5) + vec3(0, 0, 0.5), 1.0);
}
