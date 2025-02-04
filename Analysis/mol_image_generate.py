import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt
import torch

# Constants
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600
FONTSIZE = 20


def getdata(filename):
    """Loads data from an npz file."""
    data = np.load(filename, allow_pickle=True)
    return data["arr_0"] if "arr_0" in data else data


def to_superscript(number):
    """Convert a number to its Unicode superscript representation."""
    superscript_map = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
    }
    return "".join(superscript_map.get(char, char) for char in str(number))


def restore_node_feats(transformed):
    """Restores node features based on a predefined mapping."""
    if isinstance(transformed, np.ndarray):
        transformed = torch.tensor(transformed, dtype=torch.float32)

    restored = torch.full_like(transformed, -2, dtype=torch.float32)

    mapping = {
        1: 6, 20: 8, 10: 7, 5: 17, 15: 9, 8: 35, 3: 16, 12: 15,
        18: 1, 2: 5, 16: 53, 4: 14, 6: 34, 7: 19, 9: 11, 11: 3,
        13: 30, 14: 33, 17: 12, 19: 52
    }

    for key, val in mapping.items():
        restored[:, 0] = torch.where(transformed[:, 0] == key, val, restored[:, 0])

    return restored.numpy()


def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, bond_to_edge, bond_orders, classes):
    """Visualizes molecules with correct bond orders from bond data."""

    n_components, labels = Chem.GetMolFrags(Chem.RWMol(), asMols=False, sanitizeFrags=False)

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC
    }

    images = []
    for i in range(n_components):
        component_indices = np.where(labels == i)[0]
        mol_features = feature_matrix[component_indices]

        mol = Chem.RWMol()
        atom_labels = {}

        # Add atoms
        for idx, features in zip(component_indices, mol_features):
            atomic_num = int(features[0])
            atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
            class_label = classes[idx] if idx < len(classes) else "?"
            atom_labels[
                atom_idx] = f"{Chem.GetPeriodicTable().GetElementSymbol(atomic_num)}{to_superscript(class_label)}"

        # Add bonds with correct bond order
        for bond_idx, (atom1, atom2) in enumerate(bond_to_edge.T):
            if atom1 in component_indices and atom2 in component_indices:
                bond_order = int(bond_orders[bond_idx]) if bond_idx < len(bond_orders) else 1
                bond_type = bond_type_map.get(bond_order, Chem.BondType.SINGLE)
                mol.AddBond(int(atom1), int(atom2), bond_type)

        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)

        # Draw molecule with labels
        img = Draw.MolToImage(mol, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
        images.append(img)

    # Display images
    for i, img in enumerate(images):
        plt.figure()
        plt.title(f"Molecule {i + 1}")
        plt.imshow(img)
        plt.axis("off")
    plt.show()


def main():
    PATH = "/Users/taka/Documents/vqgraph_0204/"
    EPOCH = 2
    adj_file = f"{PATH}/sample_adj_{EPOCH}.npz"
    feat_file = f"{PATH}/sample_node_feat_{EPOCH}.npz"
    bond_file = f"{PATH}/sample_bond_order_{EPOCH}.npz"
    bond_to_edge_file = f"{PATH}/sample_bond_to_edge_2.npz"

    arr_adj = getdata(adj_file)
    arr_feat = getdata(feat_file)
    arr_bond = getdata(bond_file)
    arr_bond_to_edge = getdata(bond_to_edge_file)

    arr_feat = restore_node_feats(arr_feat)

    # Ensure bond arrays match the expected dimensions
    if arr_bond_to_edge.shape[1] != arr_bond.shape[0]:
        raise ValueError("Mismatch between bond-to-edge mapping and bond order array sizes.")

    subset_adj_matrix = arr_adj[:200, :200]
    subset_feat_matrix = arr_feat[:200]
    subset_bond_to_edge = arr_bond_to_edge[:, :]
    subset_bond_orders = arr_bond[:200]

    classes = list(range(subset_adj_matrix.shape[0]))  # Example classes
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_feat_matrix, subset_bond_to_edge,
                                              subset_bond_orders, classes)


if __name__ == '__main__':
    main()
