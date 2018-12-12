#include "math.h"

// infinite reverse-z matrix; 
// https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
mat4x4 PerspectiveProjection(float fovY, float aspect, float nearz) {
    float f = 1.0f / tan(fovY/ 2.0f);
    return mat4x4(f / aspect, 0.0f,  0.0f,  0.0f,
                        0.0f,    f,  0.0f,  0.0f,
                        0.0f, 0.0f,  0.0f,  1.0f,
                        0.0f, 0.0f, nearz,  0.0f);
}

vec4 NormalizePlane(const vec4& plane) {
    vec3 normal = vec3(plane);
    float normalize_constat = glm::dot(normal, normal);
    return plane * glm::inversesqrt(normalize_constat);
}

// infinite reverse-z matrix returns far-plan in nan value
std::array<vec4, 6> GetFrustum(const mat4x4& matrix, float max_dist) {
    mat4x4 transposed = glm::transpose(matrix);
    std::array<vec4, 6> frustums;
    frustums[0] = NormalizePlane(transposed[3] + transposed[0]); // x + w > 0
    frustums[1] = NormalizePlane(transposed[3] - transposed[0]); // x - w < 0
    frustums[2] = NormalizePlane(transposed[3] + transposed[1]); // y + w > 0
    frustums[3] = NormalizePlane(transposed[3] - transposed[1]); // y - w < 0
    frustums[4] = NormalizePlane(transposed[3] - transposed[2]); // z - w < 0
    frustums[5] = NormalizePlane(vec4(0, 0, -1, max_dist)); // infinite far plane, manual plane
    return frustums;
}
