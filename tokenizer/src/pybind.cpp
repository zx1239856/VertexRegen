#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.h"
#include "simplify.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace vr_tokenizer::cgal;

PYBIND11_MODULE(_vertexregen_tokenizer_pybind, m)
{
    m.doc() = "Python binding of VertexRegen tokenizer";
    m.def(
        "edge_collapse_with_record",
        &edge_collapse_with_record,
        "Simplify a triangle mesh with edge collapse and vertex split sequence",
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("target_number_of_vertices"),
        py::arg("target_number_of_triangles"),
        py::arg("no_placement") = false,
        py::arg("sharp_angle_threshold") = -1,
        py::arg("strict") = false,
        py::arg("record_full_info") = false);

    m.def(
        "vertex_split",
        &vertex_split,
        "Vertex split a triangle mesh",
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("v_s"),
        py::arg("v_l"),
        py::arg("v_r"),
        py::arg("v_t"));

    py::class_<PolygonSoup>(m, "PolygonSoup")
        .def(py::init<>()) // Default constructor
        .def_readonly("vertices", &PolygonSoup::vertices)
        .def_readonly("faces", &PolygonSoup::faces);

    py::class_<CollapseInfo>(m, "CollapseInfo")
        .def(py::init<>()) // Default constructor
        .def_readonly("v_s", &CollapseInfo::v_s)
        .def_readonly("v_t", &CollapseInfo::v_t)
        .def_readonly("v_s_p", &CollapseInfo::v_s_p)
        .def_readonly("v_t_p", &CollapseInfo::v_t_p)
        .def_readonly("v_placement", &CollapseInfo::v_placement)
        .def_readonly("v_l", &CollapseInfo::v_l)
        .def_readonly("v_r", &CollapseInfo::v_r)
        .def_readonly("v_l_p", &CollapseInfo::v_l_p)
        .def_readonly("v_r_p", &CollapseInfo::v_r_p)
        .def_readonly("dist", &CollapseInfo::dist)
        .def_readonly("collapsed_mesh", &CollapseInfo::collapsed_mesh);

    py::class_<Stats>(m, "Stats")
        .def(py::init<>()) // Default constructor
        .def_readonly("cleaned_mesh", &Stats::cleaned_mesh)
        .def_readonly("is_valid", &Stats::is_valid)
        .def_readonly("collected", &Stats::collected)
        .def_readonly("processed", &Stats::processed)
        .def_readonly("collapsed", &Stats::collapsed)
        .def_readonly("non_collapsable", &Stats::non_collapsable)
        .def_readonly("cost_uncomputable", &Stats::cost_uncomputable)
        .def_readonly("placement_uncomputable", &Stats::placement_uncomputable)
        .def_readonly("num_sharp_edges", &Stats::num_sharp_edges)
        .def_readonly("collapse_sequence", &Stats::collapse_sequence);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
