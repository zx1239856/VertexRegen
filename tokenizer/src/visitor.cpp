#include <CGAL/version.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include "visitor.h"
#include "mesh.h"

namespace vr_tokenizer::cgal
{
    namespace PMP = CGAL::Polygon_mesh_processing;

    Eigen::Vector3d point_to_vec(const Point_3 &p)
    {
        return {p.x(), p.y(), p.z()};
    }

#define TAG CGAL::Parallel_if_available_tag

    StatsVisitor::StatsVisitor(Stats *s, const Surface_mesh &m, const bool &r)
        : stats(s), mesh(m), record_full_info(r)
    {
    }

    void StatsVisitor::OnCollapsing(const Profile &profile, const opt::optional<Point_3> &placement)
    {
        if (!placement)
        {
            ++(stats->placement_uncomputable);
        }
        // We are collapse v0 (v_t) -> v1 (v_s)
        const auto &p0 = profile.p0();
        const auto &p1 = profile.p1();
        const auto &current_mesh = profile.surface_mesh();
        CollapseInfo info = {
            profile.v1().idx(),                                                                                                               // v_s
            profile.v0().idx(),                                                                                                               // v_t
            point_to_vec(p1),                                                                                                                 // v_s_p
            point_to_vec(p0),                                                                                                                 // v_t_p
            point_to_vec(placement ? *placement : p1),                                                                                        // v_placement
            profile.left_face_exists() ? std::make_optional<std::size_t>(profile.vL().idx()) : std::nullopt,                                  // v_l
            profile.right_face_exists() ? std::make_optional<std::size_t>(profile.vR().idx()) : std::nullopt,                                 // v_r
            profile.left_face_exists() ? std::make_optional<Eigen::Vector3d>(point_to_vec(current_mesh.point(profile.vL()))) : std::nullopt,  // v_l_p
            profile.right_face_exists() ? std::make_optional<Eigen::Vector3d>(point_to_vec(current_mesh.point(profile.vR()))) : std::nullopt, // v_r_p
            0.0,                                                                                                                              // dist
            std::nullopt                                                                                                                      // collapsed_mesh
        };
        stats->collapse_sequence.emplace_back(info);
    }

    void StatsVisitor::OnCollapsed(const Profile &profile, const vertex_descriptor &)
    {
        ++(stats->collapsed);
        if (record_full_info)
        {
            auto &info = stats->collapse_sequence.back();
            const auto &current_mesh = profile.surface_mesh();
            info.collapsed_mesh = mesh_to_polygon_soup(current_mesh);
        }
    }

} // namespace vr_tokenizer::cgal