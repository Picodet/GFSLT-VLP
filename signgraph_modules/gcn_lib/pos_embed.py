import numpy as np


def get_2d_relative_pos_embed(embed_dim, grid_size):
    coords = np.stack(np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij"), axis=-1)
    coords = coords.reshape(-1, 2)
    rel = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((rel ** 2).sum(axis=-1))
    dist = dist / max(float(dist.max()), 1.0)
    return np.repeat(dist[None, :, :], embed_dim, axis=0)
