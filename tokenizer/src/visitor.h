#pragma once

#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h>
#include "common.h"

namespace vr_tokenizer::cgal
{

#if defined(CGAL_VERSION_MAJOR) && (CGAL_VERSION_MAJOR >= 6)
#include <optional>
    namespace opt = std;
#else
    namespace opt = boost;
#endif

    struct StatsVisitor : SMS::Edge_collapse_visitor_base<Surface_mesh>
    {
        StatsVisitor(Stats *stats, const Surface_mesh &mesh, const bool &record_full_info);

        void OnCollected(const Profile &, const opt::optional<double> &)
        {
            ++(stats->collected);
        }

        void OnSelected(const Profile &, const opt::optional<double> &cost, const std::size_t &, const std::size_t &)
        {
            ++(stats->processed);
            if (!cost)
            {
                ++(stats->cost_uncomputable);
            }
        }

        void OnCollapsing(const Profile &profile, const opt::optional<Point_3> &placement);

        void OnNonCollapsable(const Profile &)
        {
            ++(stats->non_collapsable);
        }

        void OnCollapsed(const Profile &profile, const vertex_descriptor &);

        Stats *stats;
        const Surface_mesh &mesh;
        const bool record_full_info;
    };

} // namespace vr_tokenizer::cgal