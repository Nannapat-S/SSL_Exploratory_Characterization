import torch
from torch_geometric.data import Data
from src.utils.edge_generator import generate_edge

class Augmentation():
    def __init__(self, config: dict, generator: torch.Generator = None):
        self.config_patch = config["patch"]
        self.config_data = config["data_initialization"]
        self.generator = generator
    
    def __call__(self, data):
        if not self.config_patch.get("random_crop"):
            return data

        patch_size = self.config_patch.get("patch_size")

        if isinstance(patch_size, list):
            patch_idx = torch.randint(
                0,
                len(patch_size),
                (1,),
                generator=self.generator,
            ).item()
            patch_size = patch_size[patch_idx]

        data = random_crop(
            data,
            patch_size,
            generator=self.generator,
        )

        crystal = self.config_data.get("crystal")

        if crystal != "fcc":
            raise ValueError(
                f"Unsupported crystal structure: {crystal}. "
                "This augmentation currently supports FCC configurations only."
            )

        coordination_number = 12
        num_elements = self.config_data.get("num_elements")

        return data


def random_crop(
    data: Data,
    patch_size: int,
    seed_idx: int = None,
    generator: torch.Generator = None,
) -> Data:

    pos = data.pos
    num_nodes = pos.size(0)

    if seed_idx is None:
        seed_idx = torch.randint(
            0,
            num_nodes,
            (1,),
            generator=generator,
        ).item()

    center = pos[seed_idx]
    half_width = patch_size // 2

    displacements = apply_MIC(pos, center)

    patch_mask = (
        (displacements >= -half_width)
        & (displacements < -half_width + patch_size)
    ).all(dim=1)

    selected_nodes = patch_mask.nonzero(
        as_tuple=False
    ).view(-1)

    patch_pos = displacements[selected_nodes]

    # Reconstruct the cropped region as a periodic FCC patch so that
    # every lattice site retains the 12-neighbor bulk coordination.
    patch_edge_index = torch.tensor(
        generate_edge(
            patch_size,
            patch_size,
            patch_size,
            periodic=True,
        ),
        dtype=torch.long,
    )

    patch_data = Data(
        x=data.x[selected_nodes],
        edge_index=patch_edge_index,
        pos=patch_pos,
    )

    for key, item in data:
        if key in ("x", "edge_index", "pos") or key.startswith("__"):
            continue

        setattr(patch_data, key, item)

    return patch_data

def apply_MIC(pos: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Apply the minimum image convention for a cubic periodic simulation cell.

    Assumes integer lattice coordinates spanning 0 to L - 1 along each axis.
    """
    box_length = int(pos.max().item()) + 1

    displacements = (pos - center) % box_length
    displacements = torch.where(
        displacements > box_length / 2,
        displacements - box_length,
        displacements,
    )

    return displacements
