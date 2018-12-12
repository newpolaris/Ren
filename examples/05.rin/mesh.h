#pragma once

#include <volk.h>
#include <stdint.h>
#include <vector>
#include "math.h"

struct alignas(16) MeshDraw
{
    vec3 position;
    float scale;
    quat orientation;
    vec3 center;
    float radius;
    uint32_t index_count;
    uint32_t meshlet_offset;
    uint32_t meshlet_count;
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
    uint32_t vertex_offset;
    uint32_t first_instance;
};

enum { kMeshVertices = 64 };
enum { kMeshTriangles = 126 };

struct Meshlet
{
    vec3 center;
    float radius;
    int8_t cone[4];
    uint32_t vertices[kMeshVertices]; // save reference index of global vertex index
    uint8_t indices[kMeshTriangles*3];
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct alignas(16) MeshletDraw
{
    vec3 center;
    float radius;
    uint32_t index_offset; // meshlet index's
    uint32_t index_count;
    int8_t cone[4];
};

struct Mesh
{
    vec3 center;
    float radius;
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

