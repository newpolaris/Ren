#pragma once

#include <stdint.h>
#include <vector>

struct alignas(16) MeshDraw
{
	float offset[2];
	float scale[2];
};

struct alignas(16) Vertex
{
    uint16_t x, y, z, w;
    uint8_t nx, ny, nz, nw;
    uint16_t tu, tv;
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
    // devide mehsletIndices per meshlet instance
    std::vector<std::pair<uint32_t, uint32_t>> meshlet_instances;
};

Mesh LoadMesh(const std::string& filename);
std::vector<Meshlet> BuildMeshlets(const Mesh& mesh);
void BuildMeshletCones(Mesh* mesh);
void BuildMeshletIndices(Mesh* mesh);

