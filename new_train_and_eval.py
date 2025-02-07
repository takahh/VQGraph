import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

DATAPATH = "data"

class MoleculeGraphDataset(Dataset):
    def __init__(self, adj_dir, attr_dir):
        self.adj_files = sorted(glob.glob(f"{adj_dir}/concatenated_adj_batch_*.npy"))
        self.attr_files = sorted(glob.glob(f"{attr_dir}/concatenated_attr_batch_*.npy"))
        assert len(self.adj_files) == len(self.attr_files), "Mismatch in adjacency and attribute files"

    def __len__(self):
        return len(self.adj_files)

    def __getitem__(self, idx):
        adj_matrix = np.load(self.adj_files[idx])  # Load adjacency matrix
        attr_matrix = np.load(self.attr_files[idx])  # Load atom features
        return torch.tensor(adj_matrix, dtype=torch.float32), torch.tensor(attr_matrix, dtype=torch.float32)

# Initialize dataset and dataloader
dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Iterate through batches
for adj_batch, attr_batch in dataloader:
    print("Adjacency batch shape:", adj_batch.shape)
    print("Attribute batch shape:", attr_batch.shape)
    break  # Check one batch


