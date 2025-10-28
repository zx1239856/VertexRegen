#pragma once

#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>

namespace CGAL::Surface_mesh_simplification
{

  namespace internal
  {

    template <typename TriangleMesh, typename GeomTraits>
    class Plane_quadric_calculator_no_placement : public Plane_quadric_calculator<TriangleMesh, GeomTraits>
    {
      using Col_4 = typename GarlandHeckbert_matrix_types<GeomTraits>::Col_4;
      using Mat_4 = typename GarlandHeckbert_matrix_types<GeomTraits>::Mat_4;

    public:
      Col_4 construct_optimal_point(const Mat_4 &quadric, const Col_4 &p0, const Col_4 &p1) const
      {
        Col_4 opt_pt;

        const Col_4 p1mp0 = p1 - p0;
        const FT a = (p1mp0.transpose() * quadric * p1mp0)(0, 0);
        const FT b = 2 * (p0.transpose() * quadric * p1mp0)(0, 0);

        if (is_zero(a))
        {
          if (b < 0)
          {
            opt_pt = p1;
          }
          else
          {
            opt_pt = p0;
          }
        }
        else
        {
          // Choose between p0 and p1 based on which gives lower cost
          const FT p0_cost = (p0.transpose() * quadric * p0)(0, 0);
          const FT p1_cost = (p1.transpose() * quadric * p1)(0, 0);

          if (p0_cost > p1_cost)
          {
            opt_pt = p1;
          }
          else
          {
            opt_pt = p0;
          }
        }
        return opt_pt;
      }
    };

  } // namespace internal

  template <typename TriangleMesh, typename GeomTraits>
  class GarlandHeckbert_plane_no_placement_policies : public internal::GarlandHeckbert_cost_and_placement<
                                                          internal::Plane_quadric_calculator_no_placement<TriangleMesh, GeomTraits>,
                                                          TriangleMesh,
                                                          GeomTraits>
  {
  public:
    typedef internal::Plane_quadric_calculator_no_placement<TriangleMesh, GeomTraits> Quadric_calculator;

  private:
    typedef internal::GarlandHeckbert_cost_and_placement<
        Quadric_calculator, TriangleMesh, GeomTraits>
        Base;
    typedef GarlandHeckbert_plane_no_placement_policies<TriangleMesh, GeomTraits> Self;

  public:
    typedef Self Get_cost;
    typedef Self Get_placement;

    typedef typename GeomTraits::FT FT;

  public:
    GarlandHeckbert_plane_no_placement_policies(TriangleMesh &tmesh, const FT dm = FT(100))
        : Base(tmesh, Quadric_calculator(), dm)
    {
    }

  public:
    const Get_cost &get_cost() const { return *this; }
    const Get_placement &get_placement() const { return *this; }

    using Base::operator();
  };

} // namespace CGAL::Surface_mesh_simplification
