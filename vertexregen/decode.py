import trimesh
import numpy as np
from functools import partial
from vertexregen.modeling import VertexRegenConfig
from vertexregen_tokenizer.tokenize import Decoder


def cleanup_mesh(vertices, faces, return_raw=False):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if return_raw:
        return mesh
    return mesh.vertices, mesh.faces


class SequenceDecoder:
    def __init__(self, config: VertexRegenConfig, failfast=False):
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.nil_token_id = config.nil_token_id
        self.pos_token_offset = config.pos_token_offset
        self.all_pos_tokens = np.arange(
            config.pos_token_offset, config.pos_token_offset + config.num_pos_tokens
        )
        self.failfast = failfast
        self._reset_state()

    def _reset_state(self):
        self.tokens = []
        self.position = 0
        self.state = "EXPECT_VS"
        self.result = []
        self.current_structure = {}
        self.vl_is_nil = False
        self._vsplit_decoder = None
        self._last_vsplit_success = True

    def _cut_tokens(self, tokens):
        tokens = tokens[tokens != self.config.pad_token_id]
        if tokens[0] == self.config.bos_token_id:
            tokens = tokens[1:]
        if tokens[-1] == self.config.eos_token_id:
            tokens = tokens[:-1]
        sep_token_pos = (tokens == self.config.sep_token_id).nonzero()[0]
        if len(sep_token_pos) > 0:
            base_mesh_seq = tokens[: sep_token_pos[0]]
            vsplit_seq = tokens[sep_token_pos[0] + 1 :]
        else:
            base_mesh_seq = tokens
            vsplit_seq = []
        base_mesh_seq = base_mesh_seq - self.pos_token_offset
        return base_mesh_seq, vsplit_seq

    def _cleanup_base_mesh(self, base_mesh_seq):
        # MeshXL-like base mesh decoding
        vertices = base_mesh_seq[: len(base_mesh_seq) // 9 * 9].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        init_vertices, init_faces = cleanup_mesh(vertices, faces)
        init_vertices = init_vertices.astype(int)
        return init_vertices, init_faces

    def peek(self, count=1):
        chunk = self.tokens[self.position : self.position + count]
        if count == 1:
            return chunk[0] if len(chunk) > 0 else None
        return chunk

    def can_consume(self, count=1):
        return self.position + count <= len(self.tokens)

    def consume(self, count=1):
        if self.position + count > len(self.tokens):
            raise ValueError(f"Unexpected end of stream at index {self.position}")

        chunk = self.tokens[self.position : self.position + count]
        self.position += count
        return chunk[0] if count == 1 else chunk

    def init_state_with_tokens(self, tokens):
        tokens = np.array(tokens)
        self._reset_state()

        base_mesh_seq, vsplit_seq = self._cut_tokens(tokens)
        init_vertices, init_faces = self._cleanup_base_mesh(base_mesh_seq)
        self.result = [(init_vertices, init_faces)]
        self._vsplit_decoder = Decoder(init_vertices, init_faces)

        self.tokens = vsplit_seq

    def _state_transition(self):
        match self.state:
            case "EXPECT_VS":
                self._handle_vs()
            case "EXPECT_VL":
                self._handle_vl()
            case "EXPECT_VR":
                self._handle_vr()
            case "EXPECT_VT":
                self._handle_vt()
            case _:
                raise ValueError(f"Unknown state: {self.state}")

    def run(self, tokens):
        self.init_state_with_tokens(tokens)

        while self.position < len(self.tokens):
            self._state_transition()
            if not self._last_vsplit_success and self.failfast:
                break

        # Check if we ended in the middle of a structure
        if self.state != "EXPECT_VS":
            raise ValueError(
                "Stream ended incomplete. Waiting for remaining components."
            )

        return self.result

    def _get_valid_vertex_tokens(self, existing_prefix, valid_vertices):
        match len(existing_prefix):
            case 0:
                ret = np.unique(valid_vertices[:, 0])
            case 1:
                valid_mask = valid_vertices[:, 0] == existing_prefix[0]
                ret = np.unique(valid_vertices[valid_mask][:, 1])
            case 2:
                valid_mask = (valid_vertices[:, 0] == existing_prefix[0]) & (
                    valid_vertices[:, 1] == existing_prefix[1]
                )
                ret = np.unique(valid_vertices[valid_mask][:, 2])
            case 3:
                raise ValueError("Unexpected state")
        ret = ret + self.pos_token_offset
        return ret

    def _get_relevant_vertices(self, vertex, exclude_vertices=None):
        if exclude_vertices is None:
            exclude_vertices = []
        curr_vertices = self._vsplit_decoder.curr_vertices
        curr_faces = self._vsplit_decoder.curr_faces
        vertex_map = self._vsplit_decoder.vertex_map
        v_idx = vertex_map[tuple(vertex)]
        exclude_v_indices = set(vertex_map[tuple(v)] for v in exclude_vertices)
        rel_faces = np.any(curr_faces == v_idx, axis=1)
        connected_vertices = np.unique(curr_faces[rel_faces])
        connected_vertices = connected_vertices[connected_vertices != v_idx]
        connected_vertices = connected_vertices[
            ~np.isin(connected_vertices, list(exclude_v_indices))
        ]
        return curr_vertices[connected_vertices]

    def step_with_vsplit_token(self, token):
        self.tokens = np.append(self.tokens, token)
        try:
            self._state_transition()
        except ValueError:
            # Ignore error if tokens are not long enough to consume
            pass

    def get_valid_tokens(self):
        prefix = self.peek(3) - self.pos_token_offset
        match self.state:
            case "EXPECT_VS":
                if not self._last_vsplit_success and self.failfast:
                    valid_tokens = [self.eos_token_id]
                else:
                    valid_tokens = self._get_valid_vertex_tokens(
                        prefix, self._vsplit_decoder.curr_vertices
                    )
                    if len(self.result) > 1 and len(prefix) == 0:
                        # We have at least one vsplit already, allow EOS
                        valid_tokens = np.append(valid_tokens, self.eos_token_id)
            case "EXPECT_VL":
                rel_vertices = self._get_relevant_vertices(
                    self.current_structure["v_s"]
                )
                valid_tokens = self._get_valid_vertex_tokens(prefix, rel_vertices)
                if len(prefix) == 0:
                    valid_tokens = np.append(valid_tokens, self.nil_token_id)
            case "EXPECT_VR":
                v_l = self.current_structure["v_l"]
                rel_vertices = self._get_relevant_vertices(
                    self.current_structure["v_s"],
                    exclude_vertices=[v_l] if v_l is not None else None,
                )
                valid_tokens = self._get_valid_vertex_tokens(prefix, rel_vertices)
                if len(prefix) == 0 and not self.vl_is_nil:
                    valid_tokens = np.append(valid_tokens, self.nil_token_id)
            case "EXPECT_VT":
                valid_tokens = self.all_pos_tokens
            case _:
                raise ValueError(f"Unknown state: {self.state}")
        return valid_tokens

    def _handle_vs(self):
        val = self.consume(3) - self.pos_token_offset
        self.current_structure = {"v_s": val}
        self.state = "EXPECT_VL"

    def _handle_vl(self):
        token = self.peek()

        if token == self.nil_token_id:
            self.consume(1)
            val = None
            self.vl_is_nil = True
        else:
            val = self.consume(3) - self.pos_token_offset
            self.vl_is_nil = False

        self.current_structure["v_l"] = val
        self.state = "EXPECT_VR"

    def _handle_vr(self):
        token = self.peek()

        if token == self.nil_token_id:
            if self.vl_is_nil:
                raise ValueError("Invalid Sequence: v_l and v_r cannot both be <nil>")
            self.consume(1)
            val = None
        else:
            val = self.consume(3) - self.pos_token_offset

        self.current_structure["v_r"] = val
        self.state = "EXPECT_VT"

    def _handle_vt(self):
        val = self.consume(3) - self.pos_token_offset
        self.current_structure["v_t"] = val

        v_s_p = self.current_structure["v_s"]
        v_l_p = self.current_structure["v_l"]
        v_r_p = self.current_structure["v_r"]
        v_t_p = self.current_structure["v_t"]

        success = self._vsplit_decoder.apply_vsplit(
            v_s_p=v_s_p,
            v_l_p=v_l_p,
            v_r_p=v_r_p,
            v_t_p=v_t_p,
        )
        self._last_vsplit_success = success
        if success:
            vertices = self._vsplit_decoder.curr_vertices.copy()
            faces = self._vsplit_decoder.curr_faces.copy()
            self.result.append((vertices, faces))

        # Reset for next structure
        self.current_structure = {}
        self.vl_is_nil = False
        self.state = "EXPECT_VS"


def get_prefix_allowed_tokens_fn(config: VertexRegenConfig, batch_size: int = 1):
    pad_token_id = config.pad_token_id
    eos_token_id = config.eos_token_id
    sep_token_id = config.sep_token_id
    pos_token_list = np.arange(
        config.pos_token_offset, config.pos_token_offset + config.num_pos_tokens
    ).tolist()

    def _prefix_allowed_tokens(batch_id, input_ids, states):
        if input_ids[-1] == eos_token_id:
            states[batch_id]["ended"] = True
        if states[batch_id]["ended"]:
            return [pad_token_id]
        if not states[batch_id]["has_base_mesh"]:
            if input_ids[-1] == sep_token_id:
                states[batch_id]["has_base_mesh"] = True
                decoder = SequenceDecoder(config, failfast=True)
                decoder.init_state_with_tokens(input_ids.cpu())
                states[batch_id]["decoder"] = decoder
                return decoder.get_valid_tokens()
            return pos_token_list + [sep_token_id, eos_token_id]
        else:
            decoder = states[batch_id]["decoder"]
            decoder.step_with_vsplit_token(input_ids[-1].cpu())
            return decoder.get_valid_tokens()

    states = [{"has_base_mesh": False, "ended": False} for _ in range(batch_size)]
    return partial(_prefix_allowed_tokens, states=states)
