import datasets
import numpy as np
from vertexregen_tokenizer import quantized_edge_collapse
from vertexregen_tokenizer.collapse import (
    validate_vertex_split_sequence,
    validate_edge_collapse_sequence,
)


def create_vertex_split_dataset(example):
    results = {
        "uid": [],
        "vertices": [],
        "init_vertices": [],
        "init_faces": [],
        "vsplit_seq": [],
    }
    for uid, vertices, faces in zip(
        example["uid"], example["vertices"], example["faces"]
    ):
        stats = quantized_edge_collapse(vertices, faces, num_pos_tokens=128)
        if stats is None:
            continue
        assert validate_vertex_split_sequence(stats), "Vertex split sequence invalid"
    return results


data = datasets.load_dataset("zx1239856/shapenet")
data.map(
    create_vertex_split_dataset,
    batched=True,
    batch_size=8,
    remove_columns=data["train"].column_names,
)
