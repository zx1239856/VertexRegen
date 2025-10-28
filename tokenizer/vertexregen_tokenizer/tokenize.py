import numpy as np

from ._vertexregen_tokenizer_pybind import vertex_split
from .utils import sort_quantized_mesh


def tokenize_mesh(
    all_vertices,
    init_vertices,
    init_faces,
    vsplit_seq,
    bos_token_id=1,
    eos_token_id=2,
    sep_token_id=3,
    nil_token_id=4,
    pos_token_offset=5,
):
    init_vertices, init_faces = sort_quantized_mesh(init_vertices, init_faces)
    soup = init_vertices[init_faces].flatten() + pos_token_offset
    vsplit_tokens = []
    for v_s, v_l, v_r, v_t in vsplit_seq:
        v_s_p = (all_vertices[v_s] + pos_token_offset).tolist()
        v_l_p = (
            (all_vertices[v_l] + pos_token_offset).tolist()
            if v_l != -1
            else [nil_token_id]
        )
        v_r_p = (
            (all_vertices[v_r] + pos_token_offset).tolist()
            if v_r != -1
            else [nil_token_id]
        )
        v_t_p = (all_vertices[v_t] + pos_token_offset).tolist()
        vsplit_tokens.extend([*v_s_p, *v_l_p, *v_r_p, *v_t_p])
    tokens = (
        [bos_token_id] + soup.tolist() + [sep_token_id] + vsplit_tokens + [eos_token_id]
    )
    return tokens


class Decoder:
    def __init__(
        self,
        init_vertices,
        init_faces,
    ):
        self.curr_vertices = init_vertices
        self.curr_faces = init_faces
        self._update_vertex_map()

    def _update_vertex_map(self):
        self.vertex_map = {tuple(v): i for i, v in enumerate(self.curr_vertices)}

    def apply_vsplit(self, v_s_p, v_l_p, v_r_p, v_t_p):
        v_s_i = self.vertex_map.get(tuple(v_s_p), -1)
        v_l_i = self.vertex_map.get(tuple(v_l_p), -1) if v_l_p is not None else None
        v_r_i = self.vertex_map.get(tuple(v_r_p), -1) if v_r_p is not None else None
        if -1 in [v_s_i, v_l_i, v_r_i]:
            return False
        try:
            result = vertex_split(
                self.curr_vertices, self.curr_faces, v_s_i, v_l_i, v_r_i, v_t_p
            )
            if result is None:
                return False
        except RuntimeError:
            return False
        self.curr_vertices = np.array(result.vertices).astype(int)
        self.curr_faces = np.array(result.faces).astype(int)
        self._update_vertex_map()
        return True
