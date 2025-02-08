import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import copy
import torch
import dgl
from utils import set_seed
import dgl.dataloading
from train_teacher import get_args
import dgl
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import dgl
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
DATAPATH = "data/both_mono"

#                model, g,    attr_batch, optimizer, epoch, accumulation_steps
def train_sage(model, dataloader, feats, optimizer, epoch, accumulation_steps=1, lamb=1):
    model.train()
    total_loss = 0
    loss_list, latent_list = [], []
    cb_list = []
    loss_list_list = []  # Initialize a list for tracking loss_list3 over steps
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            _, logits, loss, _, cb, loss_list3, latent_train, quantized, latents = model(blocks, feats, epoch)
            loss = loss * lamb / accumulation_steps
        for i, loss_value in enumerate(loss_list3):
            loss_list_list[i].append(loss_value.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        latent_list.append(latent_train.detach().cpu())
        cb_list.append(cb.detach().cpu())
        loss_list.append(loss.detach().cpu())
        avg_loss = total_loss / len(dataloader)
        return avg_loss, loss_list_list, latent_list, latents


class MoleculeGraphDataset(Dataset):
    def __init__(self, adj_dir, attr_dir):
        self.adj_files = sorted(glob.glob(f"{adj_dir}/concatenated_adj_batch_*.npy"))
        self.attr_files = sorted(glob.glob(f"{attr_dir}/concatenated_attr_batch_*.npy"))
        assert len(self.adj_files) == len(self.attr_files), "Mismatch in adjacency and attribute files"

    def __len__(self):
        return len(self.adj_files)

    def __getitem__(self, idx):
        attr = []
        adj = []
        adj_matrix = torch.tensor(np.load(self.adj_files[idx]))  # Load adjacency matrix

        attr_matrix = torch.tensor(np.load(self.attr_files[idx]))  # Load atom features
    #     # print(f"attr_matrix.shape {attr_matrix.shape}")
    #     # pad_size = 100 - attr_matrix.shape[0]
    #     attr.append(attr_matrix)  # Pad rows only
    #     # print(f"padded_attr.shape {padded_attr.shape}")
    #
        return torch.tensor(adj_matrix, dtype=torch.float32), torch.tensor(attr_matrix, dtype=torch.float32)


def collate_fn(batch):
    """Pads adjacency matrices and attributes while handling size mismatches."""
    adj_matrices, attr_matrices = zip(*batch)

    # Find max number of nodes in this batch
    # max_nodes = max(adj.shape[0] for adj in adj_matrices)

    # # Pad adjacency matrices to ensure square shape (max_nodes, max_nodes)
    # padded_adj = []
    # for adj in adj_matrices:
    #     pad_size = max_nodes - adj.shape[0]
    #     padded_adj.append(torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size)))  # Pad both dimensions

    # padded_adj = torch.stack(padded_adj)  # Now safely stack

    # Pad attribute matrices (features) to (max_nodes, num_features)
    # num_features = attr_matrices[0].shape[1]  # Keep number of features same
    # padded_attr = []
    # for attr in attr_matrices:
    #     pad_size = max_nodes - attr.shape[0]
    #     padded_attr.append(torch.nn.functional.pad(attr, (0, 0, 0, pad_size)))  # Pad rows only
    #
    # padded_attr = torch.stack(padded_attr)  # Now safely stack

    return adj_matrices, attr_matrices


import dgl
import torch


def convert_to_dgl(adj_batch, attr_batch):
    # 細長く concat されてる行列をひとつずつ dgl のグラフにし、dgl object のリストを返す
    """Converts a batch of adjacency matrices (torch tensors) and attributes to a list of DGLGraphs."""
    graphs = []
    for i in range(len(adj_batch)):  # Loop over each molecule set (1000 molecules)
        adj_matrix = adj_batch[i]  # (100, 100)
        attr_matrix = attr_batch[i]  # (100, 100)
        adj_matrix = adj_matrix.view(1000, 100, 100)
        attr_matrix = attr_matrix.view(1000, 100, 7)
        # Ensure adjacency matrix is square
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            print(f"⚠️ Skipping non-square adjacency matrix at index {i}")
            continue
        # Identify non-zero feature vectors
        nonzero_mask = (attr_matrix.abs().sum(dim=1) > 0)  # True for nodes with non-zero features
        num_total_nodes = nonzero_mask.sum().item()  # Count non-zero feature vectors

        # Extract only the relevant feature vectors
        filtered_attr_matrix = attr_matrix[nonzero_mask]

        # Extract edges as before
        src, dst = adj_matrix.nonzero(as_tuple=True)

        # Ensure DGLGraph includes only nodes with non-zero features
        g = dgl.graph((src, dst), num_nodes=num_total_nodes)
        # Assign filtered node features
        g.ndata["feat"] = filtered_attr_matrix
        if g.num_nodes() != num_total_nodes:
            print(f"g.num_nodes() {g.num_nodes()}!= num_total_nodes {num_total_nodes}")

        # --------------------------------
        # check if the cutoff was correct
        # --------------------------------
        remaining_features = attr_matrix[g.num_nodes():]
        # Check if all values are zero
        if torch.all(remaining_features == 0):
            pass
        else:
            print("⚠️ WARNING: Non-zero values found in remaining features!")
        graphs.append(g)

    return graphs  # Return a list of graphs instead of a single one


def run_inductive(
        conf,
        model,
        optimizer,
        accumulation_steps=1
):
    # Initialize dataset and dataloader
    dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    for epoch in range(1, conf["max_epoch"] + 1):
        # --------------------------------
        # run only in train mode
        # --------------------------------
        if conf["train_or_infer"] == "train":

            # Iterate through batches
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 8:
                    break
                print(f"------{idx}-------")
                print("Adjacency batch:", adj_batch)
                print("Attribute batch:", attr_batch)
                g = convert_to_dgl(adj_batch, attr_batch)
                print(f"Graph Edge Types: {g.etypes}")
                print(f"Graph Node Types: {g.ntypes}")

                loss, loss_list_list, latent_train, latents = train_sage(
                    model, g, attr_batch, optimizer, epoch, accumulation_steps
                )
                model.encoder.reset_kmeans()
                # cb_new = model.encoder.vq._codebook.init_embed_(latents)
                # save codebook and vectors every epoch
                # cb_just_trained = np.concatenate([a.cpu().detach().numpy() for a in cb_just_trained[-1]])
                # np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())
                # latents = torch.squeeze(latents)
                # # random_indices = np.random.choice(latent_train.shape[0], 20000, replace=False)
                # np.savez(f"./latents_{epoch}", latents.cpu().detach().numpy())

