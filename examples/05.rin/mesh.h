#pragma once

#include <volk.h>
#include <stdint.h>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/quaternion_float.hpp>

struct PushConstant
{
    glm::mat4x4 project;
};

struct alignas(16) MeshDraw
{
    glm::vec3 position;
    float scale;
    glm::quat orientation;
    uint32_t index_count;
};

struct alignas(16) Vertex
{
    float x, y, z;
    uint8_t nx, ny, nz, nw;
    uint16_t tu, tv;
};

struct MeshDrawCommand {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
};

enum { kMeshVertices = 64 };
enum { kMeshTriangles = 126 };

struct Meshlet
{
    float cone[4];
    uint32_t vertices[kMeshVertices]; // save reference index of global vertex index
    uint8_t indices[kMeshTriangles * 3];
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Meshlet> meshlets;
    // meshlet indices for rendering mesh in standard vertex shader; just pre interpreted meshlet index look up
    std::vector<uint32_t> meshlet_indices;
    // devide meshlet_indices per meshlet instance
    std::vector<std::pair<uint32_t, uint32_t>> meshlet_instances;
};

Mesh LoadMesh(const std::string& filename);
std::vector<Meshlet> BuildMeshlets(const Mesh& mesh);
void BuildMeshletIndices(Mesh* mesh);

