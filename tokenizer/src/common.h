#pragma once

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <Eigen/Dense>
#include <vector>

using Kernel = CGAL::Simple_cartesian<double>;
using FT = Kernel::FT;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using halfedge_descriptor = boost::graph_traits<Surface_mesh>::halfedge_descriptor;
using edge_descriptor = boost::graph_traits<Surface_mesh>::edge_descriptor;
using vertex_descriptor = boost::graph_traits<Surface_mesh>::vertex_descriptor;
namespace SMS = CGAL::Surface_mesh_simplification;

namespace vr_tokenizer::cgal
{
    struct PolygonSoup
    {
        Eigen::ArrayX3d vertices;
        Eigen::ArrayX3i faces;
    };

    struct CollapseInfo
    {
        std::size_t v_s;
        std::size_t v_t;
        Eigen::Vector3d v_s_p;
        Eigen::Vector3d v_t_p;
        Eigen::Vector3d v_placement;
        std::optional<std::size_t> v_l;
        std::optional<std::size_t> v_r;
        std::optional<Eigen::Vector3d> v_l_p;
        std::optional<Eigen::Vector3d> v_r_p;
        double dist;
        std::optional<PolygonSoup> collapsed_mesh;
    };

    struct Stats
    {
        PolygonSoup cleaned_mesh;
        bool is_valid = false;
        size_t collected = 0;
        size_t processed = 0;
        size_t collapsed = 0;
        size_t non_collapsable = 0;
        size_t cost_uncomputable = 0;
        size_t placement_uncomputable = 0;
        size_t num_sharp_edges = 0;
        std::vector<CollapseInfo> collapse_sequence;
    };

} // namespace vr_tokenizer::cgal