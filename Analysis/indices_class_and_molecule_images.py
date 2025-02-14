import numpy as np
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
# np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from rdkit.Geometry import Point2D

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 1300
FONTSIZE = 40
EPOCH = 6
PATH = "/Users/taka/Documents/vqgraph_0213/"

def getdata(filename):
    # filename = "out_emb_list.npz"
    if "mol" in filename:
        arr = np.load(f"{filename}")
    else:
        arr = np.load(f"{filename}")["arr_0"]
    # arr = np.squeeze(arr)
    return arr

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

def compute_molecule_bounds(mol):
    """Calculate the bounding box for a molecule."""
    try:
        conformer = mol.GetConformer()  # Attempt to get the conformer
    except ValueError:
        # Fallback: Generate 2D coordinates and retry
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    minv = Point2D(min(x_coords), min(y_coords))
    maxv = Point2D(max(x_coords), max(y_coords))
    return minv, maxv


def to_superscript(number):
    """Convert a number to its Unicode superscript representation."""
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    return "".join(superscript_map.get(char, char) for char in str(number))


def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, classes, arr_src, arr_dst, arr_bond_order):
    """
    Visualizes molecules with node classes shown near the atoms.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Combined adjacency matrix for all molecules.
        feature_matrix (numpy.ndarray): Node feature matrix. First column is atomic numbers.
        classes (numpy.ndarray): Class labels for each node.
        arr_src (numpy.ndarray): Source indices of bonds.
        arr_dst (numpy.ndarray): Destination indices of bonds.
        arr_bond_order (numpy.ndarray): Bond order array.

    Returns:
        None: Displays molecule images with annotated classes near atoms.
    """
    node_indices = np.arange(feature_matrix.shape[0])  # Ensures indices match feature matrix
    node_to_class = {node: cls for node, cls in zip(node_indices, classes)}

    # Step 2: Identify connected components (molecules)
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

    images = []
    for i in range(n_components - 2):
        if i == 0:
            continue
        print(f"$$$$$$$$$$$$$$$$$$$. {i}")

        # Get node indices for this molecule
        component_indices = np.where(labels == i)[0]
        print(f"component_indices {component_indices}")

        # Extract subgraph
        mol_adj = adj_matrix[component_indices, :][:, component_indices]
        mol_features = feature_matrix[component_indices]

        # Filter edges to only those within the component
        mask = np.isin(arr_src, component_indices) & np.isin(arr_dst, component_indices)
        mol_src = arr_src[mask]
        mol_dst = arr_dst[mask]
        mol_bond = arr_bond_order[mask]

        print("mol_src", mol_src)
        print("mol_dst", mol_dst)
        print("mol_bond", mol_bond)

        # Create RDKit molecule
        mol = Chem.RWMol()
        atom_mapping = {}  # Maps original indices to RDKit indices

        # Add atoms and annotate classes
        atom_labels = {}
        for idx, features in zip(component_indices, mol_features):
            atomic_num = int(features[0])  # First element is the atomic number
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_mapping[idx] = atom_idx  # Map original index to new RDKit index

            # Annotate with class label
            class_label = node_to_class.get(idx, "Unknown")
            atom_labels[
                atom_idx] = f"{Chem.GetPeriodicTable().GetElementSymbol(atomic_num)}{class_label}" if class_label != "Unknown" else Chem.GetPeriodicTable().GetElementSymbol(
                atomic_num)

        # Map bond order
        bond_type_map = {1: Chem.BondType.SINGLE,
                         2: Chem.BondType.DOUBLE,
                         3: Chem.BondType.TRIPLE,
                         4: Chem.BondType.AROMATIC}

        for src, dst, bond_order in zip(mol_src, mol_dst, mol_bond):
            src, dst, bond_order = int(src), int(dst), int(bond_order)

            # Check if atom indices are valid
            if src not in atom_mapping or dst not in atom_mapping:
                print(f"Skipping bond ({src}, {dst}) - Atoms not found in mapping")
                continue

            src_mol, dst_mol = atom_mapping[src], atom_mapping[dst]

            # Avoid self-bonds
            if src_mol == dst_mol:
                print(f"Skipping self-bond ({src_mol}, {dst_mol})")
                continue

                # Avoid duplicate bonds
            if mol.GetBondBetweenAtoms(src_mol, dst_mol) is None:
                bond_type = bond_type_map.get(bond_order, Chem.BondType.SINGLE)

                print(f"Adding bond: {src_mol} - {dst_mol} (Bond type: {bond_type})")

                mol.AddBond(src_mol, dst_mol, bond_type)

                # **Mark atoms and bonds as aromatic if needed**
                if bond_order == 4:  # Aromatic bond
                    mol.GetAtomWithIdx(src_mol).SetIsAromatic(True)
                    mol.GetAtomWithIdx(dst_mol).SetIsAromatic(True)
                    mol.GetBondBetweenAtoms(src_mol, dst_mol).SetIsAromatic(True)
            else:
                print(f"Skipping duplicate bond: ({src_mol}, {dst_mol})")

        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Exception as e:
            print(f"Sanitization warning: {e}")

        # Draw the molecule
        drawer = Draw.MolDraw2DCairo(1500, 1000)  # Adjusted canvas size
        options = drawer.drawOptions()
        options.atomLabelFontSize = 4  # Increase font size for better readability

        for idx, label in atom_labels.items():
            options.atomLabels[idx] = label  # Assign custom labels to atoms

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert binary image data to an image
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(drawer.GetDrawingText()))
        images.append(img)
    print(images)
    # Step 4: Display images
    for i, img in enumerate(images):
        plt.figure(dpi=250)
        plt.title(f"Molecule {i + 1}")
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


import torch
import torch
import numpy as np

def restore_node_feats(transformed):
    # Convert to PyTorch tensor if it's a NumPy array
    if isinstance(transformed, np.ndarray):
        transformed = torch.tensor(transformed, dtype=torch.float32)

    restored = torch.empty_like(transformed, dtype=torch.float32)

    restored[:, 0] = torch.where(transformed[:, 0] == 1, 6,
                      torch.where(transformed[:, 0] == 20, 8,
                      torch.where(transformed[:, 0] == 10, 7,
                      torch.where(transformed[:, 0] == 5, 17,
                      torch.where(transformed[:, 0] == 15, 9,
                      torch.where(transformed[:, 0] == 8, 35,
                      torch.where(transformed[:, 0] == 3, 16,
                      torch.where(transformed[:, 0] == 12, 15,
                      torch.where(transformed[:, 0] == 18, 1,
                      torch.where(transformed[:, 0] == 2, 5,
                      torch.where(transformed[:, 0] == 16, 53,
                      torch.where(transformed[:, 0] == 4, 14,
                      torch.where(transformed[:, 0] == 6, 34,
                      torch.where(transformed[:, 0] == 7, 19,
                      torch.where(transformed[:, 0] == 9, 11,
                      torch.where(transformed[:, 0] == 11, 3,
                      torch.where(transformed[:, 0] == 13, 30,
                      torch.where(transformed[:, 0] == 14, 33,
                      torch.where(transformed[:, 0] == 17, 12,
                      torch.where(transformed[:, 0] == 19, 52, -2))))))))))))))))))))

    restored[:, 1] = torch.where(transformed[:, 1] == 1, 1,
                      torch.where(transformed[:, 1] == 20, 2,
                      torch.where(transformed[:, 1] == 10, 3,
                      torch.where(transformed[:, 1] == 15, 0,
                      torch.where(transformed[:, 1] == 5, 4,
                      torch.where(transformed[:, 1] == 7, 6,
                      torch.where(transformed[:, 1] == 12, 5, -2)))))))

    restored[:, 2] = torch.where(transformed[:, 2] == 1, 0,
                      torch.where(transformed[:, 2] == 20, 1,
                      torch.where(transformed[:, 2] == 10, -1,
                      torch.where(transformed[:, 2] == 5, 3,
                      torch.where(transformed[:, 2] == 15, 2, -2)))))

    restored[:, 3] = torch.where(transformed[:, 3] == 1, 4,
                      torch.where(transformed[:, 3] == 20, 3,
                      torch.where(transformed[:, 3] == 10, 1,
                      torch.where(transformed[:, 3] == 5, 2,
                      torch.where(transformed[:, 3] == 15, 7,
                      torch.where(transformed[:, 3] == 18, 6, -2))))))

    restored[:, 4] = torch.where(transformed[:, 4] == 1, 0,
                      torch.where(transformed[:, 4] == 20, 1, -2))

    restored[:, 5] = torch.where(transformed[:, 5] == 1, 0,
                      torch.where(transformed[:, 5] == 20, 1, -2))

    restored[:, 6] = torch.where(transformed[:, 6] == 1, 3,
                      torch.where(transformed[:, 6] == 20, 0,
                      torch.where(transformed[:, 6] == 10, 1,
                      torch.where(transformed[:, 6] == 15, 2,
                      torch.where(transformed[:, 6] == 5, 4, -2)))))

    return restored.numpy()  # Convert back to NumPy array if needed


def main():
    path = PATH
    adj_file = f"{path}/sample_adj_{EPOCH}.npz"                     # input data
    feat_file = f"{path}sample_node_feat_{EPOCH}.npz"      # assigned code vector id
    indices_file = f"{path}sample_emb_ind_{EPOCH}.npz"
    bond_order_file = f"{path}sample_bond_num_{EPOCH}.npz"
    src_file = f"{path}sample_src_{EPOCH}.npz"
    dst_file = f"{path}sample_dst_{EPOCH}.npz"

    arr_indices = getdata(indices_file)   # indices of the input
    arr_adj = getdata(adj_file)       # assigned quantized code vec indices
    print("arr_adj.shape")
    print(arr_adj.shape)
    arr_feat = getdata(feat_file)       # assigned quantized code vec indices
    arr_feat = restore_node_feats(arr_feat)
    node_indices = [int(x) for x in arr_indices.tolist()]
    arr_src = getdata(src_file)
    arr_dst = getdata(dst_file)
    arr_bond_order = getdata(bond_order_file)
    print("arr_bond_order")
    print(arr_bond_order.shape)
    print("arr_src")
    print(arr_src.shape)
    print("arr_dst")
    print(arr_dst.shape)

    # -------------------------------------
    # rebuild attr matrix
    # -------------------------------------
    # attr_data = arr_input["attr_data"]
    # attr_indices = arr_input["attr_indices"]
    # attr_indptr = arr_input["attr_indptr"]
    # attr_shape = arr_input["attr_shape"]
    # attr_matrix = csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
    # ic(node_indices[0])
    # subset_attr_matrix = attr_matrix[node_indices[0]:node_indices[0] + 200, :].toarray()
    # subset_attr_matrix = attr_matrix.toarray()

    # -------------------------------------
    # rebuild adj matrix
    # -------------------------------------
    # Assuming you have these arrays from your input
    # adj_data = arr_input["adj_data"]
    # adj_indices = arr_input["adj_indices"]
    # adj_indptr = arr_input["adj_indptr"]
    # adj_shape = arr_input["adj_shape"]
    # Reconstruct the sparse adjacency matrix
    # adj_matrix = csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
    subset_adj_matrix = arr_adj[0:200, 0:200]
    subset_attr_matrix = arr_feat[:200]
    # -------------------------------------
    # split the matrix into molecules
    # -------------------------------------
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, node_indices, arr_src, arr_dst, arr_bond_order)


if __name__ == '__main__':
    main()