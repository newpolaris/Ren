#include "mesh.h"
#include <macro.h>
#include <algorithm>
#include <meshoptimizer.h>
#include <objparser.h>

float NegativeIndexHelper(float* src, int i, int sub)
{
    return i < 0 ? 0.f : src[i * 3 + sub];
}

float HalfToFloat(uint16_t v) {
    uint16_t sign = v >> 15;
    uint16_t exp = (v >> 10) & 31;
    uint16_t man = v & 1023;

    ASSERT(exp != 31);
    if (exp == 0)
    {
        ASSERT(man == 0);
        return 0.f;
    }
    return (sign == 0 ? 1.f : -1.f) * ldexpf(float(man + 1024) / 1024.f, exp - 15);
}

Mesh LoadMesh(const std::string& filename)
{
    ObjFile obj;
    objParseFile(obj, filename.c_str());

    size_t index_count = obj.f_size / 3;
    std::vector<Vertex> vertices(index_count);

    for (size_t i = 0; i < index_count; i++)
    {
        Vertex& v = vertices[i];

        int vi = obj.f[i * 3 + 0];
        int vti = obj.f[i * 3 + 1];
        int vni = obj.f[i * 3 + 2];

        v.x = NegativeIndexHelper(obj.v, vi, 0);
        v.y = NegativeIndexHelper(obj.v, vi, 1);
        v.z = NegativeIndexHelper(obj.v, vi, 2);
        v.nx = uint8_t(NegativeIndexHelper(obj.vn, vni, 0) * 127.f + 127.f);
        v.ny = uint8_t(NegativeIndexHelper(obj.vn, vni, 1) * 127.f + 127.f);
        v.nz = uint8_t(NegativeIndexHelper(obj.vn, vni, 2) * 127.f + 127.f);
        v.nw = 0;
        v.tu = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 0));
        v.tv = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 1));
    }

    std::vector<uint32_t> indices(index_count);

    std::vector<uint32_t> remap(index_count);
    size_t vertex_count = meshopt_generateVertexRemap(remap.data(), 0, index_count, 
                                                      vertices.data(), index_count, sizeof(Vertex));

    vertices.resize(vertex_count);

    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), index_count, sizeof(Vertex), remap.data());
    meshopt_remapIndexBuffer(indices.data(), 0, index_count, remap.data());

    meshopt_optimizeVertexCache(indices.data(), indices.data(), index_count, vertex_count);
    meshopt_optimizeVertexFetch(vertices.data(), indices.data(), index_count, 
                                vertices.data(), vertex_count, sizeof(Vertex));

    float radius = 0.f;
    glm::vec3 center(0.f);
    for (auto& v : vertices)
        center += glm::vec3(v.x, v.y, v.z);
    center /= static_cast<float>(vertices.size());
    for (auto& v : vertices)
        radius = glm::max(radius,  glm::distance(glm::vec3(v.x, v.y, v.z), center));

    return Mesh { center, radius, vertices, indices };
}

std::vector<Meshlet> BuildMeshlets(const Mesh& mesh) {
    size_t max_vertices = kMeshVertices;
    size_t max_triangles = kMeshTriangles;

    auto meshlets = std::vector<meshopt_Meshlet>(
                            meshopt_buildMeshletsBound(mesh.indices.size(), max_vertices, max_triangles));
    meshlets.resize(meshopt_buildMeshlets(meshlets.data(), mesh.indices.data(), mesh.indices.size(), 
                                          mesh.vertices.size(), max_vertices, max_triangles));

    static_assert(std::is_trivial<Meshlet>::value &&
                  std::is_standard_layout<Meshlet>::value &&
                  std::is_trivial<meshopt_Meshlet>::value &&
                  std::is_standard_layout<meshopt_Meshlet>::value, "memcpy requirement");

    std::vector<Meshlet> meshlet_result;
    for (auto& meshlet : meshlets) {
        Meshlet m = {};
        memcpy(m.vertices, meshlet.vertices, sizeof(m.vertices));
        memcpy(m.indices, meshlet.indices, sizeof(m.indices));
        m.triangle_count = meshlet.triangle_count;
        m.vertex_count = meshlet.vertex_count;

        meshopt_Bounds bounds = meshopt_computeMeshletBounds(meshlet, &mesh.vertices[0].x, mesh.vertices.size(),
                                                             sizeof(Vertex));

        m.center = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
        m.radius = bounds.radius;

        m.cone[0] = bounds.cone_axis_s8[0];
        m.cone[1] = bounds.cone_axis_s8[1];
        m.cone[2] = bounds.cone_axis_s8[2];
        m.cone[3] = bounds.cone_cutoff_s8;

        meshlet_result.push_back(m);
    }
    // TODO: 32 packing

    return std::move(meshlet_result);
}

// TODO: remove function and indirect access in vertex shader?
void BuildMeshletIndices(Mesh* mesh) {
    uint32_t cnt = 0;
    std::vector<uint32_t> meshlet_indices(mesh->indices.size());
    for (const auto& meshlet : mesh->meshlets) {
        uint32_t start = cnt;
        for (uint32_t k = 0; k < uint32_t(meshlet.triangle_count) * 3; k++)
            meshlet_indices[cnt++] = meshlet.vertices[meshlet.indices[k]];
        mesh->meshlet_instances.push_back({start, cnt - start});
    }
    mesh->meshlet_indices = meshlet_indices;
}
