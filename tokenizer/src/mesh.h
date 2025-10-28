#pragma once

#include "common.h"

namespace vr_tokenizer::cgal
{

    std::optional<Surface_mesh> polygon_soup_to_mesh(
        const Eigen::ArrayX3d &vertices,
        const Eigen::ArrayX3i &faces,
        bool strict,
        bool clean);

    PolygonSoup mesh_to_polygon_soup(const Surface_mesh &mesh);

} // namespace vr_tokenizer::cgal