#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <array>
#include <optional>
#include "mesh.h"

namespace vr_tokenizer::cgal
{
    namespace PMP = CGAL::Polygon_mesh_processing;

    using Custom_point = std::array<FT, 3>;
    using CGAL_Polygon = std::array<std::size_t, 3>;

    struct Array_traits
    {
        struct Equal_3
        {
            bool operator()(const Custom_point &p, const Custom_point &q) const
            {
                return (p == q);
            }
        };

        struct Less_xyz_3
        {
            bool operator()(const Custom_point &p, const Custom_point &q) const
            {
                return std::lexicographical_compare(p.begin(), p.end(), q.begin(), q.end());
            }
        };

        Equal_3 equal_3_object() const { return Equal_3(); }
        Less_xyz_3 less_xyz_3_object() const { return Less_xyz_3(); }
    };

    std::optional<Surface_mesh> polygon_soup_to_mesh(
        const Eigen::ArrayX3d &vertices,
        const Eigen::ArrayX3i &faces,
        bool strict,
        bool clean)
    {
        std::vector<Custom_point> points;
        points.reserve(vertices.rows());
        std::vector<CGAL_Polygon> polygons;
        polygons.reserve(faces.rows());
        for (int i = 0; i < vertices.rows(); ++i)
        {
            points.push_back({vertices(i, 0), vertices(i, 1), vertices(i, 2)});
        }
        for (int i = 0; i < faces.rows(); ++i)
        {
            polygons.push_back({static_cast<std::size_t>(faces(i, 0)),
                                static_cast<std::size_t>(faces(i, 1)),
                                static_cast<std::size_t>(faces(i, 2))});
        }
        if (clean)
        {
            PMP::repair_polygon_soup(points, polygons, CGAL::parameters::geom_traits(Array_traits()));
        }
        auto is_valid = PMP::orient_polygon_soup(points, polygons);
        if (strict && !is_valid)
        {
            return std::nullopt;
        }
        Surface_mesh mesh;
        PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
        return mesh;
    };

    PolygonSoup mesh_to_polygon_soup(const Surface_mesh &mesh)
    {
        PolygonSoup soup;
        soup.vertices.resize(mesh.number_of_vertices(), 3);
        soup.faces.resize(mesh.number_of_faces(), 3);
        std::unordered_map<vertex_descriptor, std::size_t> vtx_id_map;
        std::size_t v_idx = 0;
        for (const auto v : mesh.vertices())
        {
            const auto &p = mesh.point(v);
            soup.vertices(v_idx, 0) = p.x();
            soup.vertices(v_idx, 1) = p.y();
            soup.vertices(v_idx, 2) = p.z();
            vtx_id_map[v] = v_idx;
            ++v_idx;
        }
        std::size_t f_idx = 0;
        for (const auto f : mesh.faces())
        {
            std::size_t i = 0;
            for (const auto v : vertices_around_face(mesh.halfedge(f), mesh))
            {
                soup.faces(f_idx, i) = static_cast<int>(vtx_id_map[v]);
                ++i;
                if(i > 3) {
                    throw std::runtime_error("Non-triangular face encountered");
                }
            }
            ++f_idx;
        }
        return soup;
    }
}