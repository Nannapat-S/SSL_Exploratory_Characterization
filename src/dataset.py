import os.path as osp
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.utils.edge_generator import generate_edge

import torch
from torch_geometric.data import Dataset, Data
from torch.nn.functional import one_hot

class MCDataset(Dataset):
    """
    Dataset for Monte Carlo configurations of the CoCuFeNiZn alloy systems.

    Expected directory structure:
        root/<alloy_id>/<composition_id>/<temperature>/step<mc_step>.txt

    Raw configuration columns:
        0: x coordinate
        1: y coordinate
        2: z coordinate
        3: simulator species index

    Raw species indices are canonicalized to:
        [Ni, Zn, Co, Cu, Fe]

    Each processed configuration is stored as a PyTorch Geometric Data object.
    """
    def __init__(self, root, composition_ids, mc_step, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.composition_ids = composition_ids
        self.mc_step = mc_step
        self.file_paths = self._discover_raw_files()
        super().__init__(root, transform, pre_transform, pre_filter)

    def _discover_raw_files(self):
        file_paths = []

        for composition_id in self.composition_ids:
            pattern = osp.join(
                self.root,
                "*",  # alloy ID
                str(composition_id),
                "*",  # temperature
                f"step{self.mc_step}.txt",
            )
            file_paths.extend(glob(pattern))

        return sorted(file_paths, key=self._raw_file_sort_key)

    @staticmethod
    def _parse_metadata(raw_path):
        parts = raw_path.split(osp.sep)

        alloy_id = int(parts[-4])
        composition_id = int(parts[-3])
        temperature = int(parts[-2])

        return alloy_id, composition_id, temperature

    @staticmethod
    def _raw_file_sort_key(path):
        return MCDataset._parse_metadata(path)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed_data")

    @property
    def raw_file_names(self):
        # return [osp.basename(path) for path in self.file_paths]
        return [f"step{self.mc_step}.txt"]

    @property
    def processed_file_names(self):
        return [
            self._processed_filename(*self._parse_metadata(path))
            for path in self.file_paths
        ]

    @staticmethod
    def _processed_filename(alloy_id, composition_id, temperature):
        return f"data_{alloy_id}_{composition_id}_{temperature}.pt"

    def download(self):
        pass
    
    def process(self):
        """Process the raw atomic data into PyG format"""
        for raw_path in tqdm(self.file_paths):
            df = pd.read_csv(
                raw_path,
                sep=r"\s+",
                header=None,
            )

            alloy_id, composition_id, temperature = self._parse_metadata(raw_path)

            grid_size = round(len(df) ** (1 / 3))

            raw_species = torch.tensor(
                df.iloc[:, 3].values,
                dtype=torch.long,
            )

            x = one_hot(
                raw_species,
                num_classes=5,
            ).float()

            pos = torch.tensor(
                df.iloc[:, :3].values,
                dtype=torch.float,
            )

            edge_index = torch.tensor(
                generate_edge(
                    grid_size,
                    grid_size,
                    grid_size,
                    periodic=True,
                ),
                dtype=torch.long,
            )

            data = Data(
                x=x,
                edge_index=edge_index,
                pos=pos,
                alloy_id=alloy_id,
                composition_id=composition_id,
                temperature=temperature,
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            filename = self._processed_filename(
                alloy_id,
                composition_id,
                temperature,
            )

            torch.save(
                data,
                osp.join(self.processed_dir, filename),
            )

    def __len__(self):
        return len(self.file_paths)

    
    def __getitem__(self, idx):
        path = osp.join(
            self.processed_dir,
            self.processed_file_names[idx],
        )

        data = torch.load(
            path,
            weights_only=False,
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
    
CANONICAL_SPECIES = ("Ni", "Zn", "Co", "Cu", "Fe")

# For each alloy ID, entry i gives the canonical species index
# corresponding to raw simulator type i.
# Canonical species order: [Ni, Zn, Co, Cu, Fe].
BASE_RAW_TO_CANONICAL = {
    1:  [0, 1, 2, 3, 4],  # Ni Zn Co Cu Fe
    2:  [4, 1, 2, 3, 0],  # Fe Zn Co Cu Ni
    3:  [4, 0, 2, 3, 1],  # Fe Ni Co Cu Zn
    4:  [3, 1, 2, 4, 0],  # Cu Zn Co Fe Ni
    5:  [3, 0, 2, 4, 1],  # Cu Ni Co Fe Zn
    6:  [3, 4, 2, 0, 1],  # Cu Fe Co Ni Zn
    7:  [2, 1, 3, 4, 0],  # Co Zn Cu Fe Ni
    8:  [2, 0, 3, 4, 1],  # Co Ni Cu Fe Zn
    9:  [2, 4, 3, 0, 1],  # Co Fe Cu Ni Zn
    10: [2, 3, 4, 0, 1],  # Co Cu Fe Ni Zn
}

COMPOSITION_GROUP_0_TO_1 = {1, 7, 12, 16, 19}
COMPOSITION_GROUP_BASE = {2, 3, 4, 5, 8, 9, 10, 13, 14, 17}
COMPOSITION_GROUP_1_TO_0 = {6, 11, 15, 18, 20}

def _get_species_mapping(alloy_id, composition_id):
    """Return raw-species-index -> canonical-species-index mapping."""

    if alloy_id not in BASE_RAW_TO_CANONICAL:
        raise ValueError(f"Unsupported alloy ID: {alloy_id}")

    mapping = BASE_RAW_TO_CANONICAL[alloy_id].copy()

    if composition_id in COMPOSITION_GROUP_0_TO_1:
        # Raw type 0 represents the same chemical species as raw type 1.
        mapping[0] = mapping[1]

    elif composition_id in COMPOSITION_GROUP_BASE:
        pass

    elif composition_id in COMPOSITION_GROUP_1_TO_0:
        # Raw type 1 represents the same chemical species as raw type 0.
        mapping[1] = mapping[0]

    elif composition_id == 21:
        # Raw types 0 and 1 both become the species represented by raw type 2.
        mapping[0] = mapping[2]
        mapping[1] = mapping[2]

    else:
        raise ValueError(f"Unsupported composition ID: {composition_id}")

    return mapping

def canonicalize_species_channels(data):
    """Canonicalize node species channels to [Ni, Zn, Co, Cu, Fe]."""

    if hasattr(data, "alloy_id") and hasattr(data, "composition_id"):
        alloy_id = int(data.alloy_id)
        composition_id = int(data.composition_id)
    else:
        # Backward compatibility with previously processed graphs.
        alloy_id, composition_id = map(int, data.id.split("_"))

    mapping = _get_species_mapping(alloy_id, composition_id)

    raw_species = data.x.argmax(dim=1)

    mapping_tensor = torch.tensor(
        mapping,
        dtype=torch.long,
        device=data.x.device,
    )

    canonical_species = mapping_tensor[raw_species]

    data.x = one_hot(
        canonical_species,
        num_classes=len(CANONICAL_SPECIES),
    ).float()

    return data
