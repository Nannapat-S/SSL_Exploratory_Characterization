import numpy as np

FCC_NEAREST_NEIGHBOR_OFFSETS = np.array(
    [
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (0, -1, 0),
    (-1, -1, 0),
    (-1, 0, 0),
    (0, 0, -1),
    (0, -1, -1),
    (-1, -1, -1),
    ],
    dtype=int
)

def generate_edge(Lx, Ly, Lz, periodic=True):
    atom_indices = np.arange(
        Lx * Ly * Lz
    ).reshape(Lx, Ly, Lz)

    i, j, k = np.meshgrid(
        np.arange(Lx),
        np.arange(Ly),
        np.arange(Lz),
        indexing="ij",
    )

    i = i.ravel()
    j = j.ravel()
    k = k.ravel()

    source_indices = atom_indices[i, j, k]

    neighbor_i = i[:, None] + FCC_NEAREST_NEIGHBOR_OFFSETS[:, 0]
    neighbor_j = j[:, None] + FCC_NEAREST_NEIGHBOR_OFFSETS[:, 1]
    neighbor_k = k[:, None] + FCC_NEAREST_NEIGHBOR_OFFSETS[:, 2]

    if periodic:
        neighbor_i %= Lx
        neighbor_j %= Ly
        neighbor_k %= Lz

        destination_indices = atom_indices[
            neighbor_i,
            neighbor_j,
            neighbor_k,
        ]

        source_indices = np.repeat(
            source_indices[:, None],
            len(FCC_NEAREST_NEIGHBOR_OFFSETS),
            axis=1,
        )

        source_indices = source_indices.ravel()
        destination_indices = destination_indices.ravel()

    else:
        valid_neighbors = (
            (0 <= neighbor_i) & (neighbor_i < Lx)
            & (0 <= neighbor_j) & (neighbor_j < Ly)
            & (0 <= neighbor_k) & (neighbor_k < Lz)
        )

        source_list = []
        destination_list = []

        for node_idx in range(len(source_indices)):
            valid = valid_neighbors[node_idx]

            ni = neighbor_i[node_idx][valid]
            nj = neighbor_j[node_idx][valid]
            nk = neighbor_k[node_idx][valid]

            destinations = atom_indices[ni, nj, nk]

            sources = np.full_like(
                destinations,
                source_indices[node_idx],
            )

            source_list.append(sources)
            destination_list.append(destinations)

        source_indices = np.concatenate(source_list)
        destination_indices = np.concatenate(destination_list)

    edge_index = np.vstack([
        source_indices,
        destination_indices,
    ])

    return edge_index
