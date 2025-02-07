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

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    """Pads adjacency matrices and attributes to the size of the largest molecule in the batch."""
    adj_matrices, attr_matrices = zip(*batch)

    # Find max nodes in batch
    max_nodes = max(adj.shape[0] for adj in adj_matrices)

    # Pad adjacency matrices
    padded_adj = [torch.nn.functional.pad(adj, (0, max_nodes - adj.shape[0], 0, max_nodes - adj.shape[1])) for adj in adj_matrices]
    padded_adj = pad_sequence(padded_adj, batch_first=True)  # Now safely stack

    # Pad attribute matrices
    padded_attr = [torch.nn.functional.pad(attr, (0, 0, 0, max_nodes - attr.shape[0])) for attr in attr_matrices]
    padded_attr = pad_sequence(padded_attr, batch_first=True)  # Now safely stack

    return padded_adj, padded_attr

# Initialize dataset and dataloader
dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Iterate through batches
for adj_batch, attr_batch in dataloader:
    print(f"-------------")
    print("Adjacency batch shape:", adj_batch.shape)
    print("Attribute batch shape:", attr_batch.shape)


