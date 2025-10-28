import numpy as np


def normalize_vertices(vertices, bound=1.0):
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    vertices = vertices - center
    scale = 2 * bound / (vmax - vmin).max()
    vertices = vertices * scale
    return vertices


def quantize_points(normalized_points, num_pos_tokens):
    # [-1, 1] -> [0, 1]
    normalized_points = (normalized_points + 1) / 2
    # error bound - 1 / (2 * num_pos_tokens)
    return (
        np.floor(normalized_points * num_pos_tokens)
        .clip(0, num_pos_tokens - 1)
        .astype(np.int32)
    )


def dequantize_points(quantized_points, num_pos_tokens):
    normalized = (quantized_points + 0.5) / num_pos_tokens
    return normalized * 2 - 1


def sort_quantized_mesh(quantized_vertices, faces):
    # Y-up to Z-up
    quantized_vertices = quantized_vertices[:, [2, 0, 1]]
    sort_inds = np.lexsort(quantized_vertices.T)
    sorted_vertices = quantized_vertices[sort_inds]
    inv_inds = np.argsort(sort_inds)
    faces = inv_inds[faces]
    # sort within faces
    start_inds = faces.argmin(axis=1)
    all_inds = start_inds[:, None] + np.arange(3)[None, :]
    all_inds = all_inds % 3
    faces = np.take_along_axis(faces, all_inds, axis=1)  # [M, 3]
    # sort among faces
    face_sort_inds = np.lexsort(faces[:, ::-1].T)
    sorted_faces = faces[face_sort_inds]
    # Z-up back to Y-up
    sorted_vertices = sorted_vertices[:, [1, 2, 0]]
    return sorted_vertices, sorted_faces
