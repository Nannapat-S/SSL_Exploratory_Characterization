import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from src.models import InfoGraph
from src.dataset import MCDataset, canonicalize_species_channels
from src.train import train
from src.inference import inference
from src.utils.augmentations import Augmentation

def main(config, run_train=False, run_inference=False):
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Model initialization
    model = InfoGraph(**config["model"])

    # Instantiate dataset
    root_path = config["data"]["root_path"]
    composition_ids = config["data"]["composition_ids"]
    mc_step = config["data"]["mc_step"]

    filter_dataset = MCDataset(
        root=root_path,
        composition_ids=composition_ids,
        mc_step=mc_step,
        pre_transform=canonicalize_species_channels,
        transform=None
    )
    temp_min = config["data"]["train_temperature_min"]
    temp_max = config["data"]["train_temperature_max"]
    filtered_indices = [i for i, data in enumerate(filter_dataset) if temp_min <= data.temperature <= temp_max]

    train_dataset = MCDataset(
        root=root_path,
        composition_ids=composition_ids,
        mc_step=mc_step,
        pre_transform=canonicalize_species_channels,
        transform=Augmentation(config, generator=gen),
    )
    train_subdataset = Subset(train_dataset, filtered_indices)
    train_loader = DataLoader(
        train_subdataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=gen
    )
 
    inference_dataset = MCDataset(
        root=root_path,
        composition_ids=composition_ids,
        mc_step=mc_step,
        pre_transform=canonicalize_species_channels,
        transform=None
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False
    )
    
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    run_dir = config["paths"]["run_dir"]
    output_dir = config["paths"]["output_dir"]

    # Train model
    if run_train:
        train(
            model=model,
            loader=train_loader,
            config=config,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir
        )
    
    # Full configuration inference
    if run_inference:
        inference(model=model,
            loader=inference_loader,
            output_dir=output_dir
        )

if __name__ == "__main__":
    import os
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    main(config=config, run_train=True, run_inference=True)
