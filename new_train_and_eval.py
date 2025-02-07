import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

DATAPATH = "data/both_mono"

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


# def collate_fn(batch):
#     """Custom collate function to pad variable-sized tensors."""
#     adj_matrices, attr_matrices = zip(*batch)
#
#     # Find max number of nodes in batch
#     max_nodes = max(adj.shape[0] for adj in adj_matrices)
#
#     # Pad adjacency matrices
#     padded_adj = [torch.nn.functional.pad(adj, (0, max_nodes - adj.shape[0], 0, max_nodes - adj.shape[1])) for adj in adj_matrices]
#     padded_adj = torch.stack(padded_adj)
#
#     # Pad attribute matrices
#     padded_attr = [torch.nn.functional.pad(attr, (0, 0, 0, max_nodes - attr.shape[0])) for attr in attr_matrices]
#     padded_attr = torch.stack(padded_attr)
#
#     return padded_adj, padded_attr
import torch
import torch

def collate_fn(batch):
    """Pads adjacency matrices and attributes while handling size mismatches."""
    try:
        adj_matrices, attr_matrices = zip(*batch)

        # Find max number of nodes in this batch
        max_nodes = max(adj.shape[0] for adj in adj_matrices)

        # Pad adjacency matrices to ensure square shape (max_nodes, max_nodes)
        padded_adj = []
        for adj in adj_matrices:
            pad_size = max_nodes - adj.shape[0]
            padded_adj.append(torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size)))  # Pad both dimensions

        padded_adj = torch.stack(padded_adj)  # Now safely stack

        # Pad attribute matrices (features) to (max_nodes, num_features)
        num_features = attr_matrices[0].shape[1]  # Keep number of features same
        padded_attr = []
        for attr in attr_matrices:
            pad_size = max_nodes - attr.shape[0]
            padded_attr.append(torch.nn.functional.pad(attr, (0, 0, 0, pad_size)))  # Pad rows only

        padded_attr = torch.stack(padded_attr)  # Now safely stack

        return padded_adj, padded_attr

    except RuntimeError as e:
        return (None, None)  # Skip batch



# Initialize dataset and dataloader
dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Iterate through batches
for idx, (adj_batch, attr_batch) in enumerate(dataloader):
    if idx == 12:
        break
    print(f"------{idx}-------")
    print("Adjacency batch shape:", adj_batch.shape)
    print("Attribute batch shape:", attr_batch.shape)


