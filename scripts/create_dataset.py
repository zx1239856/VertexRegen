import argparse
import datasets
import numpy as np
from vertexregen_tokenizer import quantized_edge_collapse
from vertexregen_tokenizer.collapse import validate_vertex_split_sequence
from .utils import load_dataset


def create_vertex_split_dataset(examples, no_validation=False, num_pos_tokens=128):
    results = {
        "uid": [],
        "vertices": [],
        "faces": [],
        "init_vertices": [],
        "init_faces": [],
        "vsplit_seq": [],
    }
    for uid, vertices, faces in zip(
        examples["uid"], examples["vertices"], examples["faces"]
    ):
        stats = quantized_edge_collapse(vertices, faces, num_pos_tokens=num_pos_tokens)
        if stats is None:
            continue
        if not no_validation:
            valid = validate_vertex_split_sequence(stats)
            if not valid:
                continue
        if len(stats.vsplit_seq) == 0:
            continue
        results["uid"].append(uid)
        results["vertices"].append(np.array(stats.vertices).astype(int))
        results["faces"].append(np.array(stats.faces).astype(int))
        results["init_vertices"].append(np.array(stats.init_vertices).astype(int))
        results["init_faces"].append(np.array(stats.init_faces).astype(int))
        results["vsplit_seq"].append(np.array(stats.vsplit_seq).astype(int))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="zx1239856/shapenet",
        help="Path to the input dataset (HuggingFace dataset format).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the output dataset (HuggingFace dataset format).",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing the dataset.",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=32,
        help="Number of workers for data processing.",
    )
    parser.add_argument(
        "-q",
        "--num-pos-tokens",
        type=int,
        default=128,
        help="Number of position tokens for quantization.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="If set, skip validation of the vertex split sequences.",
    )
    args = parser.parse_args()

    data = load_dataset(args.input)
    result = data.map(
        create_vertex_split_dataset,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        fn_kwargs={
            "no_validation": args.no_validation,
            "num_pos_tokens": args.num_pos_tokens,
        },
        remove_columns=data["train"].column_names,
        features=datasets.Features(
            {
                "uid": datasets.Value("string"),
                "vertices": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=3)
                ),  # vertices of original mesh
                "faces": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=3)
                ),  # faces of original mesh
                "init_vertices": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=3)
                ),  # vertices of simplified mesh M_0
                "init_faces": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=3)
                ),  # faces of simplified mesh M_0
                "vsplit_seq": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=4)
                ),  # vertex split sequence to reconstruct original mesh where v_s, v_l, v_r, v_t are vertex indices w.r.t. original mesh
            }
        ),
    )
    result.save_to_disk(args.output, max_shard_size="1GB")


if __name__ == "__main__":
    main()
