import argparse
import numpy as np
import trimesh
from pathlib import Path
from vertexregen_tokenizer import tokenize_mesh
from vertexregen_tokenizer.tokenize import Decoder
from vertexregen.data import load_dataset, augment_mesh, has_duplicate_vertices
from vertexregen.config import DataConfig
from vertexregen.modeling import VertexRegenConfig
from vertexregen.decode import SequenceDecoder


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
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to process from the dataset.",
    )
    parser.add_argument(
        "-q",
        "--num-pos-tokens",
        type=int,
        default=128,
        help="Number of position tokens for quantization.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to test with data augmentation.",
    )
    out_dir = Path(parser.parse_args().output)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    data = load_dataset(args.input)[args.split]
    np.set_printoptions(threshold=30, precision=2, suppress=True)
    data_cfg = DataConfig(
        path="",
        num_pos_tokens=args.num_pos_tokens,
        random_scale=True,
        random_scale_max=1.5,
        random_shift=True,
        random_shift_min=-0.2,
        random_shift_max=0.2,
        random_rotate=True,
    )
    model_config = VertexRegenConfig(
        num_pos_tokens=args.num_pos_tokens,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=3,
        nil_token_id=4,
        pos_token_offset=5,
    )
    seq_decoder = SequenceDecoder(model_config)
    num_samples = min(args.num_samples, len(data))
    for example in data.shuffle().select(range(num_samples)):
        uid = example["uid"]
        init_vertices_ref = np.array(example["init_vertices_ref"])
        init_faces = np.array(example["init_faces"])
        vsplit_seq = np.array(example["vsplit_seq"])
        vertices = np.array(example["vertices"])
        faces = np.array(example["faces"])
        if args.augment:
            vertices, aug_info = augment_mesh(vertices, data_cfg, return_info=True)
            assert not has_duplicate_vertices(
                vertices
            ), f"Duplicate vertices after augmentation for UID: {uid}"
            aug_info_str = ", ".join([f"{k}: {v}" for k, v in aug_info.items()])
        else:
            aug_info_str = "None"
        init_vertices = vertices[init_vertices_ref]
        decoder = Decoder(init_vertices, init_faces)
        init_mesh = trimesh.Trimesh(vertices=init_vertices, faces=init_faces)
        init_mesh.export(out_dir / f"{uid}_m_0.ply")
        for v_s, v_l, v_r, v_t in vsplit_seq:
            v_l_p = vertices[v_l] if v_l != -1 else None
            v_r_p = vertices[v_r] if v_r != -1 else None
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
        gt_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
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
            f"UID: {uid}, Base mesh ratio: {base_mesh_num_tokens / len(tokens):.2f}, Tokens: {tokens}, Aug: {aug_info_str}"
        )
        decoded_results = seq_decoder.run(tokens)
        assert (
            len(decoded_results) == len(vsplit_seq) + 1
        ), f"Decoding length mismatch for UID: {uid}"
        decoded_vertices, decoded_faces = decoded_results[-1]
        assert len(decoded_vertices) == len(
            vertices
        ), f"Vertex count mismatch for UID: {uid}"
        assert len(decoded_faces) == len(faces), f"Face count mismatch for UID: {uid}"


if __name__ == "__main__":
    main()
