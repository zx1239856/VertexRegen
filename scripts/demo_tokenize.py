import argparse
import numpy as np
import trimesh
from pathlib import Path
from vertexregen_tokenizer import tokenize_mesh
from vertexregen_tokenizer.tokenize import Decoder
from .utils import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset (HuggingFace dataset format).",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., train, test, validation).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the output meshes.",
    )
    out_dir = Path(parser.parse_args().output)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    data = load_dataset(args.input)[args.split]
    np.set_printoptions(threshold=30)
    for example in data.shuffle().select(range(5)):
        uid = example["uid"]
        init_vertices = np.array(example["init_vertices"])
        init_faces = np.array(example["init_faces"])
        vsplit_seq = np.array(example["vsplit_seq"])
        vertices = np.array(example["vertices"])
        decoder = Decoder(init_vertices, init_faces)
        init_mesh = trimesh.Trimesh(vertices=init_vertices, faces=init_faces)
        init_mesh.export(out_dir / f"{uid}_m_0.ply")
        for v_s, v_l, v_r, v_t in vsplit_seq:
            v_l_p = (
                vertices[v_l] if v_l != -1 else None
            )
            v_r_p = (
                vertices[v_r] if v_r != -1 else None
            )
            success = decoder.apply_vsplit(
                v_s_p=vertices[v_s],
                v_l_p=v_l_p,
                v_r_p=v_r_p,
                v_t_p=vertices[v_t],
            )
            if not success:
                print(f"Failed to apply vertex split for UID: {uid}")
        final_mesh = trimesh.Trimesh(
            vertices=decoder.curr_vertices, faces=decoder.curr_faces
        )
        final_mesh.export(out_dir / f"{uid}_m_t.ply")
        gt_mesh = trimesh.Trimesh(vertices=vertices, faces=example["faces"])
        gt_mesh.export(out_dir / f"{uid}_m_gt.ply")
        tokens = tokenize_mesh(
            all_vertices=vertices,
            init_vertices=init_vertices,
            init_faces=init_faces,
            vsplit_seq=vsplit_seq,
            bos_token_id=1,
            eos_token_id=2,
            sep_token_id=3,
            nil_token_id=4,
            pos_token_offset=5,
        )
        tokens = np.array(tokens)
        sep_token_index = np.where(tokens == 3)[0][0]
        base_mesh_num_tokens = sep_token_index - 1  # excluding BOS
        print(
            f"UID: {uid}, Base mesh ratio: {base_mesh_num_tokens / len(tokens):.2f}, Tokens: {tokens}"
        )


if __name__ == "__main__":
    main()
