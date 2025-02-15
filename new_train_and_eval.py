import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from models import WeightedThreeHopGCN
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


def transform_node_feats(a):
    transformed = torch.empty_like(a)
    transformed[:, 0] = torch.where(a[:, 0] == 6, 1,
                        torch.where(a[:, 0] == 8, 20, torch.where(a[:, 0] == 7, 10,
                        torch.where(a[:, 0] == 17, 5, torch.where(a[:, 0] == 9, 15,
                        torch.where(a[:, 0] == 35, 8, torch.where(a[:, 0] == 16, 3,
                        torch.where(a[:, 0] == 15, 12, torch.where(a[:, 0] == 1, 18,
                        torch.where(a[:, 0] == 5, 2, torch.where(a[:, 0] == 53, 16,
                        torch.where(a[:, 0] == 14, 4, torch.where(a[:, 0] == 34, 6,
                        torch.where(a[:, 0] == 19, 7, torch.where(a[:, 0] == 11, 9,
                        torch.where(a[:, 0] == 3, 11, torch.where(a[:, 0] == 30, 13,
                        torch.where(a[:, 0] == 33, 14, torch.where(a[:, 0] == 12, 17,
                        torch.where(a[:, 0] == 52, 19, -2))))))))))))))))))))

    transformed[:, 1] = torch.where(a[:, 1] == 1, 1,
    torch.where(a[:, 1] == 2, 20, torch.where(a[:, 1] == 3, 10,
    torch.where(a[:, 1] == 0, 15, torch.where(a[:, 1] == 4, 5,
    torch.where(a[:, 1] == 6, 7,
    torch.where(a[:, 1] == 5, 12, -2)))))))

    transformed[:, 2] = torch.where(a[:, 2] == 0, 1,
    torch.where(a[:, 2] == 1, 20, torch.where(a[:, 2] == -1, 10,
    torch.where(a[:, 2] == 3, 5,
    torch.where(a[:, 2] == 2, 15, -2)))))

    transformed[:, 3] = torch.where(a[:, 3] == 4, 1,
    torch.where(a[:, 3] == 3, 20, torch.where(a[:, 3] == 1, 10,
    torch.where(a[:, 3] == 2, 5, torch.where(a[:, 3] == 7, 15,
    torch.where(a[:, 3] == 6, 18, -2))))))

    transformed[:, 4] = torch.where(a[:, 4] == 0, 1,
    torch.where(a[:, 4] == 1, 20, -2))

    transformed[:, 5] = torch.where(a[:, 5] == 0, 1,
    torch.where(a[:, 5] == 1, 20, -2))

    transformed[:, 6] = torch.where(a[:, 6] == 3, 1,
    torch.where(a[:, 6] == 0, 20, torch.where(a[:, 6] == 1, 10,
    torch.where(a[:, 6] == 2, 15, torch.where(a[:, 6] == 4, 5, -2)))))

    return transformed


def train_sage(model, g, feats, optimizer, epoch, accumulation_steps=1, lamb=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feats = feats.to(device)  # Ensure loss is also on GPU
    model.train()
    loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        _, logits, loss, _, cb, loss_list3, latent_train, quantized, latents, sample_list_train = model(g, feats, epoch) # g is blocks
    loss = loss.to(device)
    del logits, quantized
    torch.cuda.empty_cache()
    scaler.scale(loss).backward(retain_graph=False)  # Ensure this is False unless needed
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    latent_list.append(latent_train.detach().cpu())
    cb_list.append(cb.detach().cpu())
    return loss, loss_list3, latent_list, latents


def evaluate(model, g, feats, epoch, accumulation_steps=1, lamb=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feats = feats.to(device)  # Ensure feats are on GPU
    model.eval()
    loss_list, latent_list, cb_list, loss_list_list = [], [], [], []
    # with torch.no_grad(), autocast():
    with torch.no_grad():
        _, logits, test_loss, _, cb, test_loss_list3, latent_train, quantized, test_latents, sample_list_test = model(g, feats, epoch)  # g is blocks
    latent_list.append(latent_train.detach().cpu())
    cb_list.append(cb.detach().cpu())
    test_latents = test_latents.detach().cpu()
    return test_loss, test_loss_list3, latent_list, test_latents, sample_list_test


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
    # ------------------------------------------------------------------------
    # 細長く concat されてる行列をひとつずつ dgl のグラフにし、dgl object のリストを返す
    # ------------------------------------------------------------------------
    """Converts a batch of adjacency matrices (torch tensors) and attributes to a list of DGLGraphs."""
    graphs = []
    for i in range(len(adj_batch)):  # Loop over each molecule set (1000 molecules)
        adj_matrices = adj_batch[i]  # (100, 100)
        attr_matrices = attr_batch[i]  # (100, 100)
        adj_matrices = adj_matrices.view(1000, 100, 100)
        attr_matrices = attr_matrices.view(1000, 100, 7)
        for j in range(len(attr_matrices)):
            adj_matrix = adj_matrices[j]
            attr_matrix = attr_matrices[j]
            # ------------------------------------------------------------------------
            # パディングを除去するためにパディング幅を検出 : attr
            # ------------------------------------------------------------------------
            nonzero_mask = (attr_matrix.abs().sum(dim=1) > 0)  # True for nodes with non-zero features
            num_total_nodes = nonzero_mask.sum().item()  # Count non-zero feature vectors
            filtered_attr_matrix = attr_matrix[nonzero_mask]
            filtered_adj_matrix = adj_matrix[:num_total_nodes, :num_total_nodes]
            # print(f"filtered_adj_matrix　{filtered_adj_matrix}")
            # print(f"filtered_attr_matrix　{filtered_attr_matrix}")
            # ------------------------------------------------------------------------
            # ゼロパディングを抜いて、dgl graph を作成
            # ------------------------------------------------------------------------
            src, dst = filtered_adj_matrix.nonzero(as_tuple=True)
            #
            # # ------------------------------------------------------------------------
            # # 隣接情報の無いノードをチェック
            # # ------------------------------------------------------------------------
            # # Sum across each row to get the number of outgoing edges per node
            # out_degrees = filtered_adj_matrix.sum(dim=1)  # Sum along columns
            # # Identify nodes with zero outgoing edges
            # zero_out_degree_nodes = torch.where(out_degrees == 0)[0]
            # if len(zero_out_degree_nodes.tolist()) > 0:
            #     for index0 in zero_out_degree_nodes.tolist():
            #         print(f"Element {filtered_attr_matrix[index0][0]} has no edge")
            edge_weights = adj_matrix[src, dst]
            g = dgl.graph((src, dst), num_nodes=num_total_nodes)
            g = dgl.add_self_loop(g)

            # 🔵 Preserve original bond order values in edge weights
            new_src, new_dst = g.edges()
            new_edge_weights = torch.zeros(len(new_src), dtype=torch.float32)

            for idx, (s, d) in enumerate(zip(new_src, new_dst)):
                if s == d:
                    new_edge_weights[idx] = 1.0  # Assign weight 1 to self-loops
                else:
                    new_edge_weights[idx] = filtered_adj_matrix[s, d]  # Preserve the original bond order

            g.edata["weight"] = new_edge_weights
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


from torch.utils.data import Dataset
import dgl
class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs  # List of DGLGraphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]


def run_inductive(
        conf,
        model,
        optimizer,
        accumulation_steps=1
):
    import gc
    import torch
    import itertools
    # ----------------------------
    # define train and test list
    # ----------------------------
    # Initialize dataset and dataloader
    dataset = MoleculeGraphDataset(adj_dir=DATAPATH, attr_dir=DATAPATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_list_list_train = [[]] * 10
        loss_list_list_test = [[]] * 10
        loss_list = []
        print(f"epoch {epoch} ------------------------------")
        # --------------------------------
        # Train
        # --------------------------------
        if conf["train_or_infer"] == "train":
            # Iterate through batches
            for idx, (adj_batch, attr_batch) in enumerate(dataloader):
                if idx == 5:
                    break
                glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
                chunk_size = 400  # in 10,000 molecules
                for i in range(0, len(glist), chunk_size):
                    chunk = glist[i:i + chunk_size]
                    batched_graph = dgl.batch(chunk)
                    # -----------------------------------------------
                    # エッジのないノードがあるか確認
                    # -----------------------------------------------
                    # for i, g in enumerate(dgl.unbatch(batched_graph)):
                    #     zero_in_degree_nodes = torch.where(g.in_degrees() == 0)[0]
                    #     if len(zero_in_degree_nodes) > 0:
                    #         print(f"Graph {i} has zero in-degree nodes: {zero_in_degree_nodes.tolist()}")
                    #         # Convert to dense adjacency matrix
                    #         adj_matrix = g.adjacency_matrix().to_dense()
                    #         print(f"Adjacency Matrix of Graph {i}:")
                    #         print(adj_matrix)
                    #         print(f"Feature Matrix of Graph {i}:")
                    #         print(g.ndata["feat"])  # Prints the full feature matrix

                    # # Get the first graph from the batch
                    # first_graph = dgl.unbatch(batched_graph)[0]
                    # # Compute in-degrees and out-degrees
                    # in_degrees = first_graph.in_degrees()
                    # out_degrees = first_graph.out_degrees()
                    # # Find nodes with no incoming edges
                    # zero_in_degree_nodes = torch.where(in_degrees == 0)[0]
                    # print(f"Nodes with zero in-degree: {zero_in_degree_nodes.tolist()}")
                    # # Find nodes with no outgoing edges
                    # zero_out_degree_nodes = torch.where(out_degrees == 0)[0]
                    # print(f"Nodes with zero out-degree: {zero_out_degree_nodes.tolist()}")


                    # Ensure node features are correctly extracted
                    with torch.no_grad():
                        batched_feats = batched_graph.ndata["feat"]
                    # batched_feats = batched_graph.ndata["feat"]
                    loss, loss_list_train, latent_train, latents = train_sage(
                        model, batched_graph, batched_feats, optimizer, epoch, accumulation_steps)
                    model.reset_kmeans()
                    cb_new = model.vq._codebook.init_embed_(latents)
                    loss_list.append(loss.detach().cpu().item())  # Ensures loss does not retain computation graph
                    torch.cuda.synchronize()
                    del batched_graph, batched_feats, chunk
                    gc.collect()
                    torch.cuda.empty_cache()
                    np.savez(f"./init_codebook_{epoch}", cb_new.cpu().detach().numpy())
                    latents = torch.squeeze(latents)
                    # random_indices = np.random.choice(latent_train.shape[0], 20000, replace=False)
                    np.savez(f"./latents_{epoch}", latents.cpu().detach().numpy())
                    loss_list_list_train = [x + [y] for x, y in zip(loss_list_list_train, loss_list_train)]

        # --------------------------------
        # Test
        # --------------------------------
        test_loss_list = []
        for idx, (adj_batch, attr_batch) in enumerate(itertools.islice(dataloader, 10, None), start=10):
            if idx == 11:
                break
            glist = convert_to_dgl(adj_batch, attr_batch)  # 10000 molecules per glist
            chunk_size = 400  # in 10,000 molecules
            for i in range(0, len(glist), chunk_size):
                chunk = glist[i:i + chunk_size]
                batched_graph = dgl.batch(chunk)
                # Ensure node features are correctly extracted
                with torch.no_grad():
                    batched_feats = batched_graph.ndata["feat"]
                # batched_feats = batched_graph.ndata["feat"]
                test_loss, loss_list_test, latent_train, latents, sample_list_test = evaluate(
                    model, batched_graph, batched_feats, epoch)
                model.reset_kmeans()
                test_loss_list.append(test_loss.cpu().item())  # Ensures loss does not retain computation graph
                torch.cuda.synchronize()
                del batched_graph, batched_feats, chunk
                gc.collect()
                torch.cuda.empty_cache()
                loss_list_list_test = [x + [y] for x, y in zip(loss_list_list_test, loss_list_test)]

        print(f"epoch {epoch}: loss {sum(loss_list)/len(loss_list):.7f}, test_loss {sum(test_loss_list)/len(test_loss_list):.7f}")

        print(f"train - div_element_loss: {sum(loss_list_list_train[0]) / len(loss_list_list_train[0]): 7f}, "
              f"train - bond_num_div_loss: {sum(loss_list_list_train[1]) / len(loss_list_list_train[1]): 7f}, "
              f"train - aroma_div_loss: {sum(loss_list_list_train[2]) / len(loss_list_list_train[2]): 7f}, "
              f"train - ringy_div_loss: {sum(loss_list_list_train[3]) / len(loss_list_list_train[3]): 7f}, "
              f"train - h_num_div_loss: {sum(loss_list_list_train[4]) / len(loss_list_list_train[4]): 7f}, "
              f"train - elec_state_div_loss: {sum(loss_list_list_train[6]) / len(loss_list_list_train[6]): 7f}, "
              f"train - charge_div_loss: {sum(loss_list_list_train[5]) / len(loss_list_list_train[5]): 7f}, "
              f"train - sil_loss: {sum(loss_list_list_train[9]) / len(loss_list_list_train[9]): 7f}")

        print(f"test - div_element_loss: {sum(loss_list_list_test[0]) / len(loss_list_list_test[0]): 7f}, "
              f"test - bond_num_div_loss: {sum(loss_list_list_test[1]) / len(loss_list_list_test[1]): 7f}, "
              f"test - aroma_div_loss: {sum(loss_list_list_test[2]) / len(loss_list_list_test[2]): 7f}, "
              f"test - ringy_div_loss: {sum(loss_list_list_test[3]) / len(loss_list_list_test[3]): 7f}, "
              f"test - h_num_div_loss: {sum(loss_list_list_test[4]) / len(loss_list_list_test[4]): 7f}, "
              f"test - elec_state_div_loss: {sum(loss_list_list_test[6]) / len(loss_list_list_test[6]): 7f}, "
              f"test - charge_div_loss: {sum(loss_list_list_test[5]) / len(loss_list_list_test[5]): 7f}, "
              f"test - sil_loss: {sum(loss_list_list_test[9]) / len(loss_list_list_test[9]): 7f}")

        np.savez(f"./sample_emb_ind_{epoch}", sample_list_test[0].cpu())
        np.savez(f"./sample_node_feat_{epoch}", sample_list_test[1].cpu())
        np.savez(f"./sample_adj_{epoch}", sample_list_test[2].cpu()[:1000, :1000])
        np.savez(f"./sample_bond_num_{epoch}", sample_list_test[3].cpu()[:1000])
        np.savez(f"./sample_src_{epoch}", sample_list_test[4].cpu()[:1000])
        np.savez(f"./sample_dst_{epoch}", sample_list_test[5].cpu()[:1000])

