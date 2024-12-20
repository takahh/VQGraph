import numpy as np
from icecream import ic

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix

from scipy.sparse import csr_matrix


# Step 1: Extract the subgraph
def extract_subgraph(graph_matrices, indices):
    # Extract sparse adjacency matrix
    adj_matrix = csr_matrix((
        graph_matrices['adj_data'],
        graph_matrices['adj_indices'],
        graph_matrices['adj_indptr']
    ), shape=graph_matrices['adj_shape'])

    # Subset the adjacency matrix for the given indices
    sub_adj = adj_matrix[indices, :][:, indices]

    # Extract attributes
    attr_data = graph_matrices['attr_data']
    sub_attr = attr_data[indices]

    return sub_adj, sub_attr


# Step 2: Convert subgraph to molecule
def subgraph_to_molecule(sub_adj, sub_attr, class_names):
    G = nx.from_numpy_matrix(sub_adj)
    mol = Chem.RWMol()

    # Add atoms
    for i, atomic_num in enumerate(sub_attr):
        atom = Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    # Add bonds
    for i, j in zip(*np.where(sub_adj > 0)):
        if i < j:  # Avoid duplicate bonds
            mol.AddBond(i, j, Chem.BondType.SINGLE)

    return mol


def getdata(filename):
    # filename = "out_emb_list.npz"
    if "mol" in filename:
        arr = np.load(f"{filename}")
    else:
        arr = np.load(f"{filename}")["arr_0"]
    # arr = np.squeeze(arr)
    return arr


def main():
    path = "/Users/taka/Downloads/latent_and_ind/"
    input_mol_file = f"/Users/taka/Downloads/molecules.npz"
    class_file = f"{path}embed_ind_indices_first8000_6.npz"
    indices_file = f"{path}idx_test_ind_tosave_first8000_6.npz"

    arr_input = getdata(input_mol_file)
    arr_indices = getdata(indices_file)
    arr_class = getdata(class_file)
    #
    # # ic(arr_input.shape)
    # # ic(arr_input)
    # ic(arr_indices.shape)
    # ic(arr_indices)
    # ic(arr_class.shape)
    # ic(arr_class)

    test_indices = arr_indices[:5]

    # -----------------------------
    # take subgraph from indices
    # -----------------------------
    sub_adj, sub_attr = extract_subgraph(arr_input, test_indices)
    ic(sub_adj)
    ic(sub_attr)
    mol = subgraph_to_molecule(sub_adj, sub_attr, graph_matrices['class_names'])
    # Step 3: Visualize molecule
    Draw.MolToImage(mol).show()


if __name__ == '__main__':
    main()