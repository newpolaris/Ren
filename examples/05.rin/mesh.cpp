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

        v.x = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 0));
        v.y = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 1));
        v.z = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 2));
        v.w = 0;
        v.nx = uint8_t(NegativeIndexHelper(obj.vn, vni, 0) * 127.f + 127.f);
        v.ny = uint8_t(NegativeIndexHelper(obj.vn, vni, 1) * 127.f + 127.f);
        v.nz = uint8_t(NegativeIndexHelper(obj.vn, vni, 2) * 127.f + 127.f);
        v.nw = 0;
        v.tu = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 0));
        v.tv = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 1));
    }

    std::vector<uint32_t> indices(index_count);

    std::vector<uint32_t> remap(index_count);
    size_t vertex_count = meshopt_generateVertexRemap(remap.data(), 0, index_count, vertices.data(), index_count, sizeof(Vertex));

    vertices.resize(vertex_count);

    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), index_count, sizeof(Vertex), remap.data());
    meshopt_remapIndexBuffer(indices.data(), 0, index_count, remap.data());

    meshopt_optimizeVertexCache(indices.data(), indices.data(), index_count, vertex_count);
    meshopt_optimizeVertexFetch(vertices.data(), indices.data(), index_count, vertices.data(), vertex_count, sizeof(Vertex));

    return Mesh{std::move(vertices), std::move(indices)};
}

std::vector<Meshlet> BuildMeshlets(const Mesh& mesh) {
    std::vector<Meshlet> meshlets;

    Meshlet let = {};
    std::vector<uint8_t> table(mesh.vertices.size(), 0xff);

    for (uint32_t i = 0; i < mesh.indices.size(); i += 3) {

        uint32_t ai = mesh.indices[i + 0];
        uint32_t bi = mesh.indices[i + 1];
        uint32_t ci = mesh.indices[i + 2];

        uint8_t& a = table[ai];
        uint8_t& b = table[bi];
        uint8_t& c = table[ci];

        uint32_t extra = (a == 0xff) + (b == 0xff) + (c == 0xff);
        // vertex_count indicated to end()
        if (let.vertex_count + extra > kMeshVertices || let.triangle_count >= kMeshTriangles) {
            meshlets.push_back(let);

            for (size_t j = 0; j < let.vertex_count; j++)
                table[let.vertices[j]] = 0xff;
            let = {};
        }

        if (a == 0xff) {
            a = let.vertex_count;
            let.vertices[let.vertex_count++] = ai;
        }

        if (b == 0xff) {
            b = let.vertex_count;
            let.vertices[let.vertex_count++] = bi;
        }

        if (c == 0xff) {
            c = let.vertex_count;
            let.vertices[let.vertex_count++] = ci;
        }

        uint32_t index = let.triangle_count * 3;
        let.indices[index + 0] = a;
        let.indices[index + 1] = b;
        let.indices[index + 2] = c;
        let.triangle_count++;
    }
    if (let.triangle_count)
        meshlets.push_back(let);
    return std::move(meshlets);
}

void BuildMeshletCones(Mesh* mesh) {
    for (auto& meshlet : mesh->meshlets) {
        std::vector<float[3]> normals(kMeshTriangles);

        for (uint16_t i = 0; i < meshlet.triangle_count; i++) {
            auto a = meshlet.indices[i * 3 + 0];
            auto b = meshlet.indices[i * 3 + 1];
            auto c = meshlet.indices[i * 3 + 2];

            const auto& va = mesh->vertices[meshlet.vertices[a]];
            const auto& vb = mesh->vertices[meshlet.vertices[b]];
            const auto& vc = mesh->vertices[meshlet.vertices[c]];

            float p0[3] = { HalfToFloat(va.x), HalfToFloat(va.y), HalfToFloat(va.z) };
            float p1[3] = {HalfToFloat(vb.x), HalfToFloat(vb.y), HalfToFloat(vb.z)};
            float p2[3] = {HalfToFloat(vc.x), HalfToFloat(vc.y), HalfToFloat(vc.z)};

            float p10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            float p20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

            // cross(p10, p20)
            float normalx = p10[1] * p20[2] - p10[2] * p20[1];
            float normaly = p10[2] * p20[0] - p10[0] * p20[2];
            float normalz = p10[0] * p20[1] - p10[1] * p20[0];

            float area = sqrtf(normalx*normalx + normaly*normaly + normalz*normalz);
            float invarea = area == 0.f ? 0.f : 1.f / area;

            normals[i][0] = normalx * invarea;
            normals[i][1] = normaly * invarea;
            normals[i][2] = normalz * invarea;
        }

        float normal[4] = {};
        for (int i = 0; i < meshlet.triangle_count; i++)
            for (int t = 0; t < 3; t++)
                normal[t] += normals[i][t];

        for (int t = 0; t < 3; t++)
            normal[t] /= meshlet.triangle_count;

        float length = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (length <= 0.f)
        {
            normal[0] = 1.f;
            normal[1] = 0.f;
            normal[2] = 0.f;
            normal[3] = 1.f;
        }
        else
        {
            float inverseLength = 1.f / length;
            for (int t = 0; t < 3; t++)
                normal[t] *= inverseLength;

            float mindp = 1.f;
            for (int i = 0; i < meshlet.triangle_count; i++)
            {
                float dp = 0.f;
                for (int t = 0; t < 3; t++)
                    dp += normals[i][t] * normal[t];
                mindp = std::min(mindp, dp);
            }
            normal[3] = mindp <= 0.f ? 1 : sqrtf(1 - mindp * mindp);
        }
        for (int t = 0; t < 4; t++)
            meshlet.cone[t] = normal[t];
    }
}

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

#if _DEBUG
    size_t culled = 0;
    for (const Meshlet& meshlet : mesh->meshlets)
        if (meshlet.cone[2] > meshlet.cone[3])
            culled++;
    printf("Culled meshlets: %d/%d\n", int(culled), int(mesh->meshlets.size()));
#endif
}
