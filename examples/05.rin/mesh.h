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
    uint32_t vertex_offset;
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
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

enum { kMeshVertices = 64 };
enum { kMeshTriangles = 126 };

struct MeshletData
{
    uint32_t vertices[kMeshVertices]; // save reference index of global vertex index
    uint8_t indices[kMeshTriangles*3];
};

struct alignas(16) Meshlet
{
    vec3 center;
    float radius;
    uint32_t index_offset;
    int8_t cone[4];
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct Mesh
{
    vec3 center;
    float radius;
    uint32_t vertex_offset;
    uint32_t vertex_count;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t meshlet_offset;
    uint32_t meshlet_count;
};

struct Geometry 
{
    std::vector<Mesh> meshes;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Meshlet> meshlets;
    std::vector<MeshletData> meshletdata;
};

void LoadMesh(const std::string& filename, Geometry* geometry);
