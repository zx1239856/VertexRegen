#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h>
#include <CGAL/Unique_hash_map.h>
#include "common.h"
#include "garland_heckbert_no_placement.h"
#include "mesh.h"
#include "simplify.h"
#include "simplify_stop_predicate.h"
#include "visitor.h"

namespace vr_tokenizer::cgal
{

  using namespace Eigen;

  using Classic_plane = SMS::GarlandHeckbert_plane_policies<Surface_mesh, Kernel>;
  using Classic_plane_no_placement = SMS::GarlandHeckbert_plane_no_placement_policies<Surface_mesh, Kernel>;

  struct Constrained_edge_map
  {
    typedef boost::readable_property_map_tag category;
    typedef bool value_type;
    typedef bool reference;
    typedef edge_descriptor key_type;

    Constrained_edge_map(const CGAL::Unique_hash_map<key_type, bool> &aConstraints)
        : mConstraints(aConstraints)
    {
    }

    value_type operator[](const key_type &e) const { return is_constrained(e); }

    friend inline value_type get(const Constrained_edge_map &m, const key_type &k) { return m[k]; }

    bool is_constrained(const key_type &e) const { return mConstraints.is_defined(e); }

  private:
    const CGAL::Unique_hash_map<key_type, bool> &mConstraints;
  };

  bool is_border(edge_descriptor e, const Surface_mesh &sm)
  {
    return (face(halfedge(e, sm), sm) == boost::graph_traits<Surface_mesh>::null_face()) ||
           (face(opposite(halfedge(e, sm), sm), sm) == boost::graph_traits<Surface_mesh>::null_face());
  }

  Point_3 point(vertex_descriptor vd, const Surface_mesh &sm)
  {
    return get(CGAL::vertex_point, sm, vd);
  }

  template <typename GH_policies>
  Stats edge_collapse_with_record_impl(
      const ArrayX3d &vertices,
      const ArrayX3i &faces,
      std::size_t target_number_of_vertices,
      std::size_t target_number_of_triangles,
      double sharp_angle_threshold,
      bool strict,
      bool record_full_info)
  {
    Stats stats;
    auto mesh_opt = polygon_soup_to_mesh(vertices, faces, strict, true);

    bool is_valid = mesh_opt.has_value() && mesh_opt->is_valid();
    stats.is_valid = is_valid;

    // Do not operate on non-manifold meshes
    if (!is_valid)
    {
      return stats;
    }

    auto &mesh = mesh_opt.value();

    stats.cleaned_mesh = mesh_to_polygon_soup(mesh);

    SMS::Simplify_stop_predicate<Surface_mesh> stop_predicate(
        target_number_of_triangles, target_number_of_vertices);

    using GH_cost = typename GH_policies::Get_cost;
    using GH_placement = typename GH_policies::Get_placement;
    using Bounded_GH_placement = SMS::Bounded_normal_change_placement<GH_placement>;

    StatsVisitor vis(&stats, mesh, record_full_info);

    GH_policies gh_policies(mesh);
    const GH_cost &gh_cost = gh_policies.get_cost();
    const GH_placement &gh_placement = gh_policies.get_placement();
    Bounded_GH_placement bounded_gh_placement(gh_placement);

    // Possibly constraint the sharp features
    CGAL::Unique_hash_map<edge_descriptor, bool> constraint_hmap(false);
    Constrained_edge_map constraints_map(constraint_hmap);
    SMS::Constrained_placement<Bounded_GH_placement, Constrained_edge_map> constrained_placement(constraints_map, bounded_gh_placement);

    bool detect_sharp_edges = sharp_angle_threshold > 0;
    auto placement = detect_sharp_edges ? constrained_placement : bounded_gh_placement;
    if (detect_sharp_edges)
    {
      // detect sharp edges
      for (edge_descriptor ed : edges(mesh))
      {
        halfedge_descriptor hd = halfedge(ed, mesh);
        if (is_border(ed, mesh))
        {
          ++stats.num_sharp_edges;
          constraint_hmap[ed] = true;
        }
        else
        {
          double angle = CGAL::approximate_dihedral_angle(
              point(target(opposite(hd, mesh), mesh), mesh),
              point(target(hd, mesh), mesh),
              point(target(next(hd, mesh), mesh), mesh),
              point(target(next(opposite(hd, mesh), mesh), mesh), mesh));
          if (CGAL::abs(angle) < sharp_angle_threshold)
          {
            ++stats.num_sharp_edges;
            constraint_hmap[ed] = true;
          }
        }
      }
    }

    SMS::edge_collapse(
        mesh,
        stop_predicate,
        CGAL::parameters::visitor(vis)
            .edge_is_constrained_map(constraints_map)
            .get_cost(gh_cost)
            .get_placement(placement));

    return stats;
  }

  Stats edge_collapse_with_record(
      const ArrayX3d &vertices,
      const ArrayX3i &faces,
      std::size_t target_number_of_vertices,
      std::size_t target_number_of_triangles,
      bool no_placement,
      double sharp_angle_threshold,
      bool strict,
      bool record_full_info)
  {
    if (no_placement)
    {
      return edge_collapse_with_record_impl<Classic_plane_no_placement>(
          vertices,
          faces,
          target_number_of_vertices,
          target_number_of_triangles,
          sharp_angle_threshold,
          strict,
          record_full_info);
    }
    else
    {
      return edge_collapse_with_record_impl<Classic_plane>(
          vertices,
          faces,
          target_number_of_vertices,
          target_number_of_triangles,
          sharp_angle_threshold,
          strict,
          record_full_info);
    }
  }

  inline void assert_hedge_valid(halfedge_descriptor h, const Surface_mesh &mesh, const std::string &msg)
  {
    if (!h.is_valid())
    {
      throw std::runtime_error(msg);
    }
  }

  inline void assert_face_valid(halfedge_descriptor h, const Surface_mesh &mesh, const std::string &msg)
  {
    if (!mesh.face(h).is_valid())
    {
      throw std::runtime_error(msg);
    }
  }

  std::optional<PolygonSoup> vertex_split(
      const ArrayX3d &vertices,
      const ArrayX3i &faces,
      std::size_t v_s,
      std::optional<std::size_t> v_l,
      std::optional<std::size_t> v_r,
      const Vector3d &v_t)
  {
    // this takes in a valid triangle soup and split the vertices
    auto mesh_opt = polygon_soup_to_mesh(vertices, faces, true, false);
    bool is_valid = mesh_opt.has_value() && mesh_opt->is_valid();
    if ((v_l.has_value() && v_s == v_l) || (v_r.has_value() && v_s == v_r) ||
        (v_l.has_value() && v_r.has_value() && v_l == v_r))
    {
      return std::nullopt;
    }
    if (!is_valid)
    {
      return std::nullopt;
    }
    auto &mesh = mesh_opt.value();
    vertex_descriptor v_s_(v_s);
    std::optional<vertex_descriptor> v_l_(v_l);
    std::optional<vertex_descriptor> v_r_(v_r);

    auto p_t = Point_3(v_t.x(), v_t.y(), v_t.z());

    if (v_l_.has_value())
    {
      auto h2 = mesh.halfedge(*v_l_, v_s_);
      assert_hedge_valid(h2, mesh, "Invalid vL");
      if (v_r_.has_value())
      {
        auto h1 = mesh.halfedge(*v_r_, v_s_);
        assert_hedge_valid(h1, mesh, "Invalid vR");
        auto h_new = CGAL::Euler::split_vertex(h1, h2, mesh);
        auto h_new_opp = mesh.opposite(h_new);
        auto v_t_ = mesh.source(h_new);
        mesh.point(v_t_) = p_t;
        if (mesh.is_border(h_new))
        {
          CGAL::Euler::add_face(CGAL::make_array(v_t_, v_s_, *v_l_), mesh);
        }
        else
        {
          auto vLvT = mesh.halfedge(*v_l_, v_t_);
          assert_hedge_valid(vLvT, mesh, "Invalid vL, vT");
          assert_face_valid(h_new, mesh, "Invalid face at h_new after split");
          CGAL::Euler::split_face(h_new, mesh.prev(vLvT), mesh);
        }
        if (mesh.is_border(h_new_opp))
        {
          CGAL::Euler::add_face(CGAL::make_array(v_s_, v_t_, *v_r_), mesh);
        }
        else
        {
          auto vRvS = mesh.halfedge(*v_r_, v_s_);
          assert_hedge_valid(vRvS, mesh, "Invalid vR, vS");
          assert_face_valid(h_new_opp, mesh, "Invalid face at h_new_opp after split");
          CGAL::Euler::split_face(h_new_opp, mesh.prev(vRvS), mesh);
        }
      }
      else
      {
        // We use CW boundary as h1
        auto h1 = h2;
        for (; !mesh.is_border(h1); h1 = mesh.next_around_target(h1))
          ;
        if (h1 == h2)
        {
          // In this case, all neighbors are around v_t. We can just move v_s to v_t location
          // And add new face
          auto p_s = mesh.point(v_s_);
          mesh.point(v_s_) = p_t;
          auto v_new = mesh.add_vertex(p_s);
          CGAL::Euler::add_face(CGAL::make_array(*v_l_, v_s_, v_new), mesh);
        }
        else
        {
          auto h_new = CGAL::Euler::split_vertex(h1, h2, mesh);
          auto v_t_ = mesh.source(h_new);
          mesh.point(v_t_) = p_t;
          auto vLvT = mesh.halfedge(*v_l_, v_t_);
          assert_hedge_valid(vLvT, mesh, "Invalid vL, vT");
          assert_face_valid(h_new, mesh, "Invalid face at h_new after split");
          CGAL::Euler::split_face(h_new, mesh.prev(vLvT), mesh);
        }
      }
    }
    else if (v_r_.has_value())
    {
      auto h2 = mesh.halfedge(*v_r_, v_s_);
      assert_hedge_valid(h2, mesh, "Invalid vR");
      auto h1_next = mesh.opposite(h2);
      if (mesh.is_border(h1_next))
      {
        // In this case, all neighbors are around v_t. We can just move v_s to v_t location
        // And add new face
        auto p_s = mesh.point(v_s_);
        mesh.point(v_s_) = p_t;
        auto v_new = mesh.add_vertex(p_s);
        CGAL::Euler::add_face(CGAL::make_array(v_s_, *v_r_, v_new), mesh);
      }
      else
      {
        auto h1 = mesh.prev(h1_next);
        for (; !mesh.is_border(h2); h2 = mesh.next_around_target(h2));
        auto h_new = CGAL::Euler::split_vertex(h1, h2, mesh);
        auto h_new_opp = mesh.opposite(h_new);
        auto v_t_ = mesh.source(h_new);
        mesh.point(v_t_) = p_t;
        auto vTvR = mesh.halfedge(v_t_, *v_r_);
        assert_hedge_valid(vTvR, mesh, "Invalid vT, vR");
        assert_face_valid(h_new_opp, mesh, "Invalid face at h_new_opp after split");
        CGAL::Euler::split_face(vTvR, mesh.prev(h_new_opp), mesh);
      }
    }
    else
    {
      return std::nullopt;
    }

    return mesh_to_polygon_soup(mesh);
  }

} // namespace vr_tokenizer::cgal
