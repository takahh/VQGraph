import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import batch
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv
from vq import VectorQuantize
import dgl
from train_and_eval import transform_node_feats
import dgl.nn as dglnn

class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linear = nn.Linear(hidden_dim, input_dim)
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
                vq = self.linear(h)
                h_list.append(vq)
        return h_list, h


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""

class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.decoder_1 = nn.Linear(input_dim, input_dim)
        self.decoder_2 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25, use_cosine_sim = True)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node

    def forward(self, g, feats):
        h = feats
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        h_list = []
        h = self.graph_layer_1(g, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)
        quantized, _, commit_loss, dist, codebook = self.vq(h)
        quantized_edge = self.decoder_1(quantized)
        quantized_node = self.decoder_2(quantized)

        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))

        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h = self.graph_layer_2(g, quantized_edge)
        h_list.append(h)
        h = self.linear(h)
        loss = feature_rec_loss + edge_rec_loss + commit_loss
        
        return h_list, h, loss, dist, codebook, [feature_rec_loss, edge_rec_loss, commit_loss]


# def feat_elem_divergence_loss(embed_ind, atom_types, num_codebooks=1500, temperature=0.02):
#     device = embed_ind.device
#
#     # Ensure embed_ind is within valid range
#     embed_ind = torch.clamp(embed_ind, min=0, max=num_codebooks - 1).long()
#
#     # Map atom_types to sequential indices
#     unique_atom_numbers = torch.unique(atom_types, sorted=True)
#     atom_types_mapped = torch.searchsorted(unique_atom_numbers, atom_types)
#
#     # Create one-hot representations
#     embed_one_hot = torch.nn.functional.one_hot(embed_ind, num_classes=num_codebooks).float()
#     atom_type_one_hot = torch.nn.functional.one_hot(atom_types_mapped, num_classes=len(unique_atom_numbers)).float()
#
#     # Compute soft assignments
#     soft_assignments = torch.softmax(embed_one_hot / temperature, dim=-1)
#
#     # Compute co-occurrence matrix
#     co_occurrence = torch.einsum("ni,nj->ij", [soft_assignments, atom_type_one_hot])
#
#     # Normalize co-occurrence
#     co_occurrence_normalized = co_occurrence / (co_occurrence.sum(dim=1, keepdim=True) + 1e-6)
#
#     # Compute row-wise entropy
#     row_entropy = -torch.sum(co_occurrence_normalized * torch.log(co_occurrence_normalized + 1e-6), dim=1)
#
#     # Compute sparsity loss
#     sparsity_loss = row_entropy.mean()
#
#     # Debug connection to the graph
#     print(f"sparsity_loss.requires_grad: {sparsity_loss.requires_grad}")
#     print(f"sparsity_loss.grad_fn: {sparsity_loss.grad_fn}")
#
#     return sparsity_loss

class SAGE(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type,
            codebook_size,
            lamb_edge,
            lamb_node,
            lamb_div_ele
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio).to(device)  # Ensure dropout is applied
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Edge encoder for bond_order
        # self.edge_encoder = nn.Linear(1, self.hidden_dim).to(device)
        self.edge_encoder = nn.Linear(1, self.hidden_dim)  # Output should be [E, hidden_dim]

        self.layers = nn.ModuleList([
            dglnn.GINEConv(
                self.edge_encoder,
                aggregator_type="sum",
                apply_func=nn.Linear(self.hidden_dim, self.hidden_dim)  # Ensure correct shape
            ).to(device) for _ in range(num_layers)
        ])
        # Optional normalization layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim).to(device) for _ in range(num_layers)
        ])

        self.linear_2 = nn.Linear(7, self.hidden_dim).to(device)  # Ensure feature transformation

        self.codebook_size = codebook_size
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, decay=0.8, use_cosine_sim=False)

        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.lamb_div_ele = lamb_div_ele

    def reset_kmeans(self):
        self.vq._codebook.reset_kmeans()

    def forward(self, blocks, feats, epoch):
        import torch
        import dgl

        # --- Preprocess Node Features ---
        # Ensure h requires gradients and apply your transformation.
        h = feats.clone() if not feats.requires_grad else feats
        # h = transform_node_feats(h)  # Your custom transformation
        init_feat = h.clone()  # Store initial features (for later use)
        torch.save(init_feat, "/h.pt")  # Save for reference

        # Get the device from h (e.g., cuda:0)
        device = h.device

        # --- Reindexing for Mini-Batch ---
        # Collect global node IDs from all blocks.
        global_node_ids = set()
        for block in blocks:
            src, dst = block.all_edges()
            global_node_ids.update(src.tolist())
            global_node_ids.update(dst.tolist())

        # Sort the global IDs to have a deterministic ordering.
        global_node_ids = sorted(global_node_ids)

        # Create a mapping: global ID -> local ID (0-indexed)
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_node_ids)}
        # print("Number of nodes in mini-batch:", len(global_to_local))
        # print("Sample mapping:", dict(list(global_to_local.items())[:5]))

        # Create an index tensor from global_node_ids on the correct device.
        idx_tensor = torch.tensor(global_node_ids, dtype=torch.int64, device=device)

        # *** Reindex the feature tensor and the initial features ***
        # This ensures both h and init_feat only have the mini-batch nodes.
        h = h[idx_tensor]
        init_feat = init_feat[idx_tensor]  # Important: reindex init_feat as well!

        # --- Remap Edge Indices and Bond Orders ---
        remapped_edge_list = []
        remapped_bond_orders = []  # List to hold bond orders, if available

        for block in blocks:
            src, dst = block.all_edges()
            src = src.to(torch.int64)
            dst = dst.to(torch.int64)

            # Remap global indices to local indices and ensure they are on the correct device.
            local_src = torch.tensor([global_to_local[i.item()] for i in src],
                                     dtype=torch.int64, device=device)
            local_dst = torch.tensor([global_to_local[i.item()] for i in dst],
                                     dtype=torch.int64, device=device)

            # Append both directions (bidirectional graph)
            remapped_edge_list.append((local_src, local_dst))
            remapped_edge_list.append((local_dst, local_src))

            # If bond orders are present in the block, remap and duplicate them.
            if "bond_order" in block.edata:
                bond_order = block.edata["bond_order"].to(torch.float32).to(device)
                remapped_bond_orders.append(bond_order)
                remapped_bond_orders.append(bond_order)  # For the reverse edge
        # --- Construct the DGL Graph ---
        # # Create a graph with nodes equal to the number of unique nodes in the mini-batch.
        # g = dgl.DGLGraph().to(device)
        # g.add_nodes(len(global_node_ids))

        edges_src = torch.cat([edge[0] for edge in remapped_edge_list])
        edges_dst = torch.cat([edge[1] for edge in remapped_edge_list])

        # Create the graph correctly
        g = dgl.graph((edges_src, edges_dst)).to(device)
        g = dgl.add_self_loop(g)  # Optional, if self-loops are needed

        # Add edges along with bond order features if available.
        if remapped_bond_orders:
            for (src, dst), bond_order in zip(remapped_edge_list, remapped_bond_orders):
                g.add_edges(src, dst, data={"bond_order": bond_order})
        else:
            for src, dst in remapped_edge_list:
                g.add_edges(src, dst)

        # Optionally add self-loops (if desired)
        g = dgl.add_self_loop(g)

        if epoch == 1:
            sample_feat = h.clone().detach()
            adj_sample = g.adjacency_matrix().to_dense()
            import sys
            torch.set_printoptions(threshold=torch.inf)  # Remove print limit

        # --- Continue with Your Forward Pass ---
        # For example, get the dense adjacency matrix.
        adj = g.adjacency_matrix().to_dense().to(device)

        h_list = []  # To store intermediate node representations

        # Example: Apply a linear transformation and the first graph layer
        h = self.linear_2(h)
        # Ensure bond_order is correctly shaped
        if "bond_order" in g.edata:
            g.edata["bond_order"] = g.edata["bond_order"].view(-1, 1).to(device)  # Ensure shape [E, 1]
            g.edata["bond_order"] = self.edge_encoder(g.edata["bond_order"])  # Transform to [E, hidden_dim]

        # Debugging print to verify fix
        print("Updated Bond order shape:", g.edata["bond_order"].shape)  # Should be [num_edges, hidden_dim]
        assert h.shape[1] == g.edata["bond_order"].shape[1], "Mismatch in feature dimensions!"
        print("Node feature shape (h)  0:", h.shape)  # Should be [num_nodes, hidden_dim]

        # Ensure everything stays on the correct device
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["bond_order"] = g.edata["bond_order"]

            # Aggregate bond order information into node features
            g.update_all(dgl.function.copy_e("bond_order", "msg"), dgl.function.mean("msg", "bond_agg"))

            # Move aggregated bond order to CUDA if needed
            g.ndata["bond_agg"] = g.ndata["bond_agg"].to(device)
            print("Node feature shape (h)  1:", h.shape)  # Should be [num_nodes, hidden_dim]

            # Concatenate aggregated bond order and node features
            h = g.ndata["h"] + g.ndata["bond_agg"]  # Ensures `h.shape = [num_nodes, 32]`
            # h = torch.cat([g.ndata["h"], g.ndata["bond_agg"]], dim=1).to(device)  # Ensure it's on CUDA

        # Node feature shape (h): torch.Size([9997, 64])
        # Bond order shape: torch.Size([282310, 32])
        print("Node feature shape (h)  2:", h.shape)  # Should be [num_nodes, hidden_dim]
        print("Bond order shape:", g.edata["bond_order"].shape)  # Should match h.shape
        assert h.shape[1] == g.edata["bond_order"].shape[1], "Mismatch in feature dimensions!"

        # Pass the correct arguments to GINEConv
        # h = self.layers(g, h, edge_feat=g.edata["bond_order"])
        # Apply all GINEConv layers sequentially
        for idx, layer in enumerate(self.layers):
            print(f"Passing h.shape: {h.shape}, bond_order.shape: {g.edata['bond_order'].shape}")
            # Passing h.shape: torch.Size([9997, 32]), bond_order.shape: torch.Size([282310, 32])
            h = layer(g, h, edge_feat=g.edata["bond_order"])
            print(f"{idx} - h.shape: {h.shape}")

        # Debugging print before passing to `GINEConv`
        print("h device:", h.device)
        print("Graph layer device:", next(self.graph_layer_1.parameters()).device)
        print("Bond order device:", g.edata["bond_order"].device)

        # Now `h` is guaranteed to be on CUDA before passing it to `GINEConv`

        # Pass correctly shaped features to GINEConv
        # h = self.graph_layer_1(g, h, edge_feat=g.edata["bond_order"])

        # Apply normalization if necessary.
        if self.norm_type != "none":
            h = self.norms[0](h)

        h_list.append(h)  # Store the latent representation

        # --- Vector Quantization Step ---
        (quantized, emb_ind, loss, dist, codebook, raw_commit_loss, latents, margin_loss,
         spread_loss, pair_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss,
         aroma_div_loss, ringy_div_loss, h_num_div_loss, sil_loss, charge_div_loss, elec_state_div_loss) = \
            self.vq(h, init_feat, epoch)

        # --- Return Outputs ---
        return (h_list, h, loss, dist, codebook,
                [div_ele_loss, bond_num_div_loss, aroma_div_loss, ringy_div_loss,
                 h_num_div_loss, charge_div_loss, elec_state_div_loss, spread_loss, pair_loss, sil_loss],
                x, detached_quantize, latents)

    def inference(self, dataloader, feats):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        dataloader : The entire graph loaded in blocks with full neighbors for each node.
        feats : The input feats of the entire node set.
        """
        device = feats.device
        dist_all = torch.zeros(feats.shape[0], self.codebook_size, device=device)
        y = torch.zeros(feats.shape[0], self.output_dim, device=device)
        latent_list = []
        input_node_list = []
        embed_ind_list = []
        div_ele_loss_list = []
        bond_num_div_loss_list = []
        aroma_div_loss_list = []
        ringy_div_loss_list = []
        h_num_div_loss_list = []
        sil_loss_list = []
        elec_state_div_loss_list = []
        charge_div_loss_list = []

        for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # Ensure features are on the correct device
            input_nodes = input_nodes.to(device)
            output_nodes = output_nodes.to(device)

            # Ensure feats requires gradients if necessary
            h = feats.clone() if not feats.requires_grad else feats
            blocks = [blk.int().to(device) for blk in blocks]

            # Get batch node features
            batch_feats = feats[input_nodes]
            batch_feats = transform_node_feats(batch_feats)

            # --- Reindexing for Mini-Batch ---
            global_node_ids = set()
            for block in blocks:
                src, dst = block.all_edges()
                global_node_ids.update(src.tolist())  # Converting to a list is okay here for set operations
                global_node_ids.update(dst.tolist())

            global_node_ids = sorted(global_node_ids)

            # Ensure valid indexing
            assert len(global_node_ids) > 0, "global_node_ids is empty!"
            assert max(global_node_ids) < feats.shape[0], "Index out of bounds in global_node_ids!"
            assert min(global_node_ids) >= 0, "Negative indices found in global_node_ids!"

            # Create a mapping: global ID -> local ID
            global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_node_ids)}

            # Create an index tensor from global_node_ids on the correct device
            idx_tensor = torch.tensor(global_node_ids, dtype=torch.int64, device=device)

            # Ensure valid feature indexing
            assert torch.max(idx_tensor) < batch_feats.shape[0], "Index out of bounds in batch_feats!"
            h = batch_feats[idx_tensor]
            init_feat = h  # Keep track of the initial features

            # --- Remap Edge Indices ---
            remapped_edge_list = []
            remapped_bond_orders = []

            for block in blocks:
                src, dst = block.all_edges()
                src, dst = src.to(torch.int64), dst.to(torch.int64)

                # Map to local IDs
                local_src = torch.tensor([global_to_local[i.item()] for i in src], dtype=torch.int64, device=device)
                local_dst = torch.tensor([global_to_local[i.item()] for i in dst], dtype=torch.int64, device=device)

                # Add bidirectional edges
                remapped_edge_list.append((local_src, local_dst))
                remapped_edge_list.append((local_dst, local_src))

                # Remap bond orders if present
                if "bond_order" in block.edata:
                    bond_order = block.edata["bond_order"].to(torch.float32).to(device)
                    remapped_bond_orders.append(bond_order)
                    remapped_bond_orders.append(bond_order)  # Bidirectional bond orders

            # --- Construct DGL Graph ---
            g = dgl.DGLGraph().to(device)
            g.add_nodes(len(global_node_ids))

            # Add edges (and bond orders if available)
            if remapped_bond_orders:
                for (src, dst), bond_order in zip(remapped_edge_list, remapped_bond_orders):
                    g.add_edges(src, dst, data={"bond_order": bond_order})
            else:
                for src, dst in remapped_edge_list:
                    g.add_edges(src, dst)

            g = dgl.add_self_loop(g)

            # Store adjacency matrix for first batch
            if idx == 0:
                # Get edge indices
                src0, dst0 = g.edges()
                # Create an empty adjacency matrix with bond orders
                adj_weighted = torch.zeros((g.num_nodes(), g.num_nodes()), device=g.device)
                # Assign bond orders to the adjacency matrix
                adj_weighted[src0, dst0] = g.edata["bond_order"].squeeze()  # Remove extra dimension if needed

                sample_feat = h.clone().detach()
                print("adj_weighted in INF")
                print(adj_weighted)
                sample_adj = adj_weighted.to_dense()

            # --- Graph Layer Processing ---
            h_list = []
            h = self.linear_2(h)
            h = self.graph_layer_1(g, h, edge_feat=g.edata["bond_order"])
            if self.norm_type != "none":
                h = self.norms[0](h)
            h_list.append(h)

            # --- Quantization ---
            (quantized, embed_ind, loss, dist, codebook, raw_commit_loss, latent_vectors, margin_loss,
             spread_loss, pair_loss, detached_quantize, x, init_cb, div_ele_loss, bond_num_div_loss, aroma_div_loss,
             ringy_div_loss, h_num_div_loss, sil_loss, charge_div_loss, elec_state_div_loss) = self.vq(h, init_feat)

            # Store computed values
            embed_ind_list.append(embed_ind)
            input_node_list.append(input_nodes)
            div_ele_loss_list.append(div_ele_loss)
            bond_num_div_loss_list.append(bond_num_div_loss)
            aroma_div_loss_list.append(aroma_div_loss)
            ringy_div_loss_list.append(ringy_div_loss)
            h_num_div_loss_list.append(h_num_div_loss)
            elec_state_div_loss_list.append(elec_state_div_loss)
            charge_div_loss_list.append(charge_div_loss)
            sil_loss_list.append(sil_loss)

            if idx == 0:
                sample_ind = embed_ind
                sample_list = [sample_ind, sample_feat, sample_adj]

        return h_list, y, loss, dist_all, codebook, [
            div_ele_loss_list, bond_num_div_loss_list, aroma_div_loss_list, ringy_div_loss_list,
            h_num_div_loss_list, charge_div_loss_list, elec_state_div_loss_list, spread_loss, pair_loss, sil_loss_list
        ], latent_list, sample_list


class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        num_heads=8,
        attn_drop=0.3,
        negative_slope=0.2,
        residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        return h_list, h


class APPNP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
        edge_drop=0.5,
        alpha=0.1,
        k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"],
                lamb_div_ele=conf["lamb_div_ele"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])

    def forward(self, data, feats, epoch):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats, epoch)

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats):
        if "SAGE" in self.model_name:
            # return self.forward(data, feats)

            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)
