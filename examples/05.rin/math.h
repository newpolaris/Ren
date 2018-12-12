#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_transform.hpp>
#include <array>

using glm::vec3;
using glm::vec4;
using glm::quat;
using glm::mat;
using glm::mat4x4;

vec4 NormalizePlane(const vec4& plane);
mat4x4 PerspectiveProjection(float fovY, float aspect, float nearz);
std::array<vec4, 6> GetFrustum(const mat4x4& matrix, float max_dist);
