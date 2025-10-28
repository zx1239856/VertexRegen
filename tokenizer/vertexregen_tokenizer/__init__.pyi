from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

__version__: str

class PolygonSoup:
    def __init__(self) -> None: ...
    vertices: NDArray[np.float64]  # shape: (N, 3)
    faces: NDArray[np.int64]  # shape: (M, 3)

class CollapseInfo:
    def __init__(self) -> None: ...
    v_s: int
    v_t: int
    v_s_p: NDArray[np.float64]  # prior position of v_s (3,)
    v_t_p: NDArray[np.float64]  # prior position of v_t (3,)
    v_placement: NDArray[np.float64]  # chosen placement (3,)
    v_l: Optional[int]
    v_r: Optional[int]
    v_l_p: Optional[NDArray[np.float64]]  # prior position of v_l (3,)
    v_r_p: Optional[NDArray[np.float64]]  # prior position of v_r (3,)
    dist: float
    collapsed_mesh: Optional[PolygonSoup]  # mesh snapshot after this collapse

class Stats:
    def __init__(self) -> None: ...
    cleaned_mesh: PolygonSoup
    is_valid: bool
    collected: int
    processed: int
    collapsed: int
    non_collapsable: int
    cost_uncomputable: int
    placement_uncomputable: int
    num_sharp_edges: int
    collapse_sequence: List[CollapseInfo]


def edge_collapse_with_record(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    target_number_of_vertices: int,
    target_number_of_triangles: int,
    no_placement: bool = False,
    sharp_angle_threshold: float = -1,
    strict: bool = False,
    record_full_info: bool = False,
) -> Stats: ...

def vertex_split(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    v_s: int,
    v_l: Optional[int],
    v_r: Optional[int],
    v_t: Sequence[float],  # expected length 3
) -> Optional[PolygonSoup]: ...
