#pragma once

#include <CGAL/license/Surface_mesh_simplification.h>

#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_profile.h>
#include <CGAL/Surface_mesh_simplification/internal/Common.h>
#include <CGAL/boost/graph/internal/helpers.h>

namespace CGAL::Surface_mesh_simplification
{

    // Stops when the number of faces falls below a given number.
    template <class TM_>
    class Simplify_stop_predicate
    {
    public:
        typedef TM_ TM;
        typedef typename boost::graph_traits<TM>::faces_size_type size_type;

        Simplify_stop_predicate(
            const std::size_t face_count_threshold,
            const std::size_t vertex_count_threshold)
            : m_face_count_threshold(face_count_threshold),
              m_vertex_count_threshold(vertex_count_threshold) {}

        template <typename F, typename Profile>
        bool operator()(
            const F & /*current_cost*/,
            const Profile &profile,
            std::size_t /*initial_edge_count*/,
            std::size_t /*current_edge_count*/) const
        {
            auto mesh = profile.surface_mesh();
            const std::size_t current_face_count =
                CGAL::internal::exact_num_faces(mesh);
            const std::size_t current_vertex_count =
                CGAL::internal::exact_num_vertices(mesh);
            return (current_face_count <= m_face_count_threshold) &&
                   (current_vertex_count <= m_vertex_count_threshold);
        }

    private:
        std::size_t m_face_count_threshold;
        std::size_t m_vertex_count_threshold;
    };

} // namespace CGAL::Surface_mesh_simplification
