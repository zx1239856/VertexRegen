import json
import datasets
import torch
import numpy as np
from pathlib import Path
from functools import partial
from transformers import PreTrainedTokenizerBase
from vertexregen_tokenizer import tokenize_mesh
from vertexregen_tokenizer.utils import quantize_points, normalize_vertices
from .config import DataConfig, ModelConfig


def pad_tokens(
    all_input_ids,
    pad_token_id,
    padding_side="right",
    max_seq_length=None,
    pad_to_multiple_of=None,
):
    max_len = max(len(input_ids) for input_ids in all_input_ids)
    if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of
    padded_input_ids = np.full(
        (len(all_input_ids), max_len), pad_token_id, dtype=np.int64
    )
    attention_mask = np.zeros((len(all_input_ids), max_len), dtype=np.int64)
    for i, input_ids in enumerate(all_input_ids):
        if padding_side == "right":
            padded_input_ids[i, : len(input_ids)] = input_ids
            attention_mask[i, : len(input_ids)] = 1
        else:
            padded_input_ids[i, -len(input_ids) :] = input_ids
            attention_mask[i, -len(input_ids) :] = 1

    if max_seq_length is not None and max_len > max_seq_length:
        padded_input_ids = padded_input_ids[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]

    return padded_input_ids, attention_mask


class VertexRegenCollator:
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig):
        self.bos_token_id = model_cfg.bos_token_id
        self.eos_token_id = model_cfg.eos_token_id
        self.sep_token_id = model_cfg.sep_token_id
        self.nil_token_id = model_cfg.nil_token_id
        self.pad_token_id = model_cfg.pad_token_id
        self.pos_token_offset = model_cfg.pos_token_offset
        self.max_seq_length = model_cfg.max_seq_length
        self.pad_to_multiple_of = data_cfg.pad_to_multiple_of
        self.padding_side = data_cfg.padding_side

    def __call__(self, examples):
        all_input_ids = []
        for example in examples:
            tokens = tokenize_mesh(
                all_vertices=example["vertices"],
                init_vertices=example["init_vertices"],
                init_faces=example["init_faces"],
                vsplit_seq=example["vsplit_seq"],
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                sep_token_id=self.sep_token_id,
                nil_token_id=self.nil_token_id,
                pos_token_offset=self.pos_token_offset,
            )
            all_input_ids.append(tokens)
        padded_input_ids, attention_mask = pad_tokens(
            all_input_ids,
            pad_token_id=self.pad_token_id,
            padding_side=self.padding_side,
            max_seq_length=self.max_seq_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = padded_input_ids.copy()
        labels[labels == self.pad_token_id] = -100
        return {
            "input_ids": torch.as_tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }


class VertexRegenTokenizer(PreTrainedTokenizerBase):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "pad_token",
        "bos_token",
        "eos_token",
        "sep_token",
        "nil_token",
        "additional_special_tokens",
    ]

    def __init__(self, model_cfg: ModelConfig):
        super().__init__(
            max_len=model_cfg.max_seq_length,
            padding_side="right",
        )
        self.is_fast = True
        self.vocab_size = model_cfg.vocab_size
        self._special_tokens_map.update(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "sep_token": "<sep>",
                "nil_token": "<nil>",
            }
        )
        self._vocab = {
            "<pad>": model_cfg.pad_token_id,
            "<bos>": model_cfg.bos_token_id,
            "<eos>": model_cfg.eos_token_id,
            "<sep>": model_cfg.sep_token_id,
            "<nil>": model_cfg.nil_token_id,
        }

    @property
    def added_tokens_decoder(self):
        return {}

    @property
    def added_tokens_encoder(self):
        return {}

    def save_vocabulary(self, save_directory, filename_prefix=None):
        out_path = Path(save_directory)
        if filename_prefix is None:
            filename_prefix = ""
        vocab_file = out_path / (filename_prefix + "vocab.json")
        with vocab_file.open("w") as fp:
            json.dump(self._vocab, fp, indent=2)
        return (str(vocab_file),)

    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            tokens = [tokens]
        ids = [self._vocab.get(token, self._vocab.get("<unk>", 0)) for token in tokens]
        if len(ids) == 1:
            return ids[0]
        return ids


def random_scale(normalized_vertices, min_scale, max_scale, info_dict=None):
    extent_max = np.abs(normalized_vertices).max(axis=0)
    extent_max[extent_max < 1e-5] = 1
    scale_limits = 1 / extent_max
    scales = np.random.uniform(min_scale, max_scale, size=3).clip(max=scale_limits)
    if info_dict is not None:
        info_dict["scales"] = scales
    scaled_vertices = normalized_vertices * scales
    return scaled_vertices


def random_shift(vertices, min_shift, max_shift, info_dict=None):
    extent_min = vertices.min(axis=0)
    extent_max = vertices.max(axis=0)
    shifts = np.random.uniform(min_shift, max_shift, size=3).clip(
        min=-1 - extent_min, max=1 - extent_max
    )
    if info_dict is not None:
        info_dict["shifts"] = shifts
    shifted_vertices = vertices + shifts
    return shifted_vertices


def random_rotate(vertices, info_dict=None):
    # random rotate along the up-axis (Y-axis)
    k = np.random.randint(0, 4)
    if info_dict is not None:
        info_dict["rotation"] = k * 90
    if k == 0:
        return vertices
    x, z = vertices[:, 0], vertices[:, 2]
    if k == 1:
        new_x, new_z = z, -x
    elif k == 2:
        new_x, new_z = -x, -z
    elif k == 3:
        new_x, new_z = -z, x
    out = np.empty_like(vertices)
    out[:, 0] = new_x
    out[:, 1] = vertices[:, 1]
    out[:, 2] = new_z
    return out


def augment_mesh(vertices, data_cfg: DataConfig, return_info=False):
    num_pos_tokens = data_cfg.num_pos_tokens
    normalized_vertices = normalize_vertices(vertices)
    has_aug = False
    info_dict = {} if return_info else None
    if data_cfg.random_scale:
        normalized_vertices = random_scale(
            normalized_vertices,
            min_scale=1.0,
            max_scale=data_cfg.random_scale_max,
            info_dict=info_dict,
        )
        has_aug = True
    if data_cfg.random_shift:
        normalized_vertices = random_shift(
            normalized_vertices,
            min_shift=data_cfg.random_shift_min,
            max_shift=data_cfg.random_shift_max,
            info_dict=info_dict,
        )
        has_aug = True
    if data_cfg.random_rotate:
        normalized_vertices = random_rotate(normalized_vertices, info_dict=info_dict)
        has_aug = True
    if has_aug:
        vertices = quantize_points(normalized_vertices, num_pos_tokens)
    if return_info:
        return vertices, info_dict
    return vertices


def has_duplicate_vertices(vertices):
    vertex_set = set()
    for v in vertices:
        v_tuple = tuple(v)
        if v_tuple in vertex_set:
            return True
        vertex_set.add(v_tuple)
    return False


def transform_mesh(examples, is_train, data_cfg: DataConfig):
    result_vertices = []
    result_init_vertices = []
    result_init_faces = []
    result_vsplit_seq = []
    for vertices, faces, init_vertices_ref, init_faces, vsplit_seq in zip(
        examples["vertices"],
        examples["faces"],
        examples["init_vertices_ref"],
        examples["init_faces"],
        examples["vsplit_seq"],
    ):
        vertices = np.array(vertices, dtype=int)
        faces = np.array(faces, dtype=int)
        init_vertices_ref = np.array(init_vertices_ref, dtype=int)
        init_faces = np.array(init_faces, dtype=int)
        vsplit_seq = np.array(vsplit_seq, dtype=int)
        if is_train:
            # Perform data augmentation
            vertices = augment_mesh(vertices, data_cfg)
        init_vertices = vertices[init_vertices_ref]
        result_vertices.append(vertices)
        result_init_vertices.append(init_vertices)
        result_init_faces.append(init_faces)
        result_vsplit_seq.append(vsplit_seq)
    return {
        "vertices": result_vertices,
        "init_vertices": result_init_vertices,
        "init_faces": result_init_faces,
        "vsplit_seq": result_vsplit_seq,
    }


def load_dataset(path):
    try:
        return datasets.load_from_disk(path)
    except:
        return datasets.load_dataset(path)


def get_dataset(data_cfg: DataConfig):
    data = load_dataset(data_cfg.path)
    train_set = data["train"]
    test_set = data["test"]
    if data_cfg.overfit:
        data_size = len(train_set)
        num_samples = min(data_size, 8)
        train_set = train_set.select(range(num_samples))
        test_set = train_set
    train_set = train_set.with_transform(
        partial(transform_mesh, is_train=not data_cfg.overfit, data_cfg=data_cfg)
    )
    test_set = test_set.with_transform(
        partial(transform_mesh, is_train=False, data_cfg=data_cfg)
    )
    return train_set, test_set
