import numpy as np
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
# np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
from icecream import ic
import matplotlib.pyplot as plt
from rdkit.Geometry import Point2D
import torch
import numpy as np
# from torch.utils.tensorboard.summary import image

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 1300
FONTSIZE = 40

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


# def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, indices_file, classes):
#     """
#     Visualizes molecules with node classes shown near the atoms.
#
#     Args:
#         adj_matrix (scipy.sparse.csr_matrix): Combined adjacency matrix for all molecules.
#         feature_matrix (numpy.ndarray): Node feature matrix. First column is atomic numbers.
#         indices_file (str): Path to file containing node indices for all atoms.
#         class_file (str): Path to file containing classes for the nodes.
#
#     Returns:
#         None: Displays molecule images with annotated classes near atoms.
#     """
#     # Step 1: Load indices and classes
#     node_indices = list(range(8000))
#     # Map node indices to classes
#     node_to_class = {node: cls for node, cls in zip(node_indices, classes)}
#
#     # Step 2: Identify connected components (molecules)
#     n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
#     print(n_components)
#     # Step 3: Extract and annotate molecules
#     images = []
#     for i in range(n_components - 2):
#         print(f"$$$$$$$$$$$$$$$$$$$. {i}")
#         # Get node indices for this molecule
#         component_indices = np.where(labels == i)[0]
#
#         # Extract subgraph
#         mol_adj = adj_matrix[component_indices, :][:, component_indices]
#         print("Adjacency Matrix:\n", mol_adj)
#
#         mol_features = feature_matrix[component_indices]
#
#         # Create RDKit molecule
#         mol = Chem.RWMol()
#
#         # Add atoms and annotate classes
#         atom_labels = {}
#         for idx, features in zip(component_indices, mol_features):
#             atomic_num = int(features[0])  # First element is the atomic number
#             atom = Chem.Atom(atomic_num)
#             atom_idx = mol.AddAtom(atom)
#             print("------")
#             print(atom_idx)
#             print(features)
#             # Annotate with superscript class label
#             class_label = node_to_class.get(idx, "Unknown")
#             if class_label != "Unknown":
#                 class_label_sup = to_superscript(class_label)
#                 atom_labels[atom_idx] = f"{Chem.GetPeriodicTable().GetElementSymbol(atomic_num)}{class_label}"
#             else:
#                 atom_labels[atom_idx] = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
#             # print(atom_labels[atom_idx])
#                 # Annotate with inline format
#
#         # # Add bonds
#         # for x, y in zip(*np.where(mol_adj > 0)):
#         #     if x < y:  # Avoid duplicate bonds
#         #         # mol.AddBond(int(x), int(y), Chem.BondType.SINGLE)
#
#         # Add bonds with correct bond order
#         bond_type_map = {1: Chem.BondType.SINGLE,
#                          2: Chem.BondType.DOUBLE,
#                          3: Chem.BondType.TRIPLE,
#                          4: Chem.BondType.AROMATIC}  # Assuming 4 means aromatic
#
#         for x, y in zip(*np.where(mol_adj > 0)):
#             if x < y:  # Avoid duplicate bonds
#                 bond_order = int(mol_adj[x, y])  # Extract bond order
#                 bond_type = bond_type_map.get(bond_order, Chem.BondType.SINGLE)
#                 mol.AddBond(int(x), int(y), bond_type)
#
#         for x, y in zip(*np.where(mol_adj > 0)):
#             if x < y:
#                 print(f"Bond {x}-{y}, Order: {mol_adj[x, y]}")
#
#         Chem.SanitizeMol(mol)  # Fixes valence issues
#         Chem.Kekulize(mol, clearAromaticFlags=True)  # Ensures explicit representation
#
#         # Compute 2D coordinates for proper display
#         AllChem.Compute2DCoords(mol)
#
#         # Ensure bonds are explicitly drawn
#         Chem.Kekulize(mol, clearAromaticFlags=True)
#
#         # Sanitize molecule
#         Chem.SanitizeMol(mol)
#         drawer = Draw.MolDraw2DCairo(CANVAS_WIDTH, CANVAS_HEIGHT)  # Increase the size (width, height)
#         options = drawer.drawOptions()
#         options.bondLineWidth = 2  # Make bonds thicker if needed
#         options.scaleBondWidth = True  # Scale bond width relative to image size
#         options.atomLabelFontSize = FONTSIZE
#         options.atomLabelPadding = 0.4  # Default is 0.2; increase for more space between labels
#
#         for idx, label in atom_labels.items():
#             options.atomLabels[idx] = label  # Assign custom labels to atoms
#
#         # Calculate and set the scale
#         minv, maxv = compute_molecule_bounds(mol)
#
#         # Set canvas dimensions
#         canvas_width = CANVAS_WIDTH
#         canvas_height = CANVAS_HEIGHT
#
#         # Calculate molecule dimensions
#         mol_width = maxv.x - minv.x
#         mol_height = maxv.y - minv.y
#
#         # Add padding
#         padding = 0.1
#         mol_width_with_padding = mol_width * (1 + padding)
#         mol_height_with_padding = mol_height * (1 + padding)
#
#         # Calculate scale
#         scale_x = canvas_width / mol_width_with_padding
#         scale_y = canvas_height / mol_height_with_padding
#         scale = min(scale_x, scale_y)
#
#         # Center the molecule
#         center_x = (canvas_width / scale - mol_width) / 2 - minv.x
#         center_y = (canvas_height / scale - mol_height) / 2 - minv.y
#
#         # Center the molecule (convert to integers for SetOffset)
#         center_x = int((canvas_width / scale - mol_width) / 2 - minv.x)
#         center_y = int((canvas_height / scale - mol_height) / 2 - minv.y)
#
#         drawer.SetOffset(center_x, center_y)
#
#         # Draw the molecule
#         drawer.DrawMolecule(mol)
#         drawer.FinishDrawing()
#
#         # Get the binary image data
#         img_data = drawer.GetDrawingText()
#
#         # Convert binary image data to an image
#         from PIL import Image
#         from io import BytesIO
#
#         img = Image.open(BytesIO(img_data))
#         images.append(img)
#
#     # Step 4: Display images
#     for i, img in enumerate(images):
#         plt.figure(dpi=150)
#         plt.title(f"Molecule {i+1}")
#         plt.imshow(img)
#         plt.axis("off")
#     plt.tight_layout()  # Automatically adjust spacing
#     plt.show()
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw



def adjacency_to_molecule(node_features, adj_matrix):
    """
    Convert an adjacency matrix and node features into an RDKit molecule.
    """
    image_list = []
    mol = Chem.RWMol()
    atom_map = {}  # Maps graph indices to RDKit atom indices

    # Step 1: Add atoms
    for i, features in enumerate(node_features):
        atomic_num = int(features[0])  # Atomic number
        formal_charge = int(features[2])  # Formal charge
        is_aromatic = bool(features[4])  # Aromaticity
        in_ring = bool(features[5])  # Ring membership

        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(formal_charge)

        mol_idx = mol.AddAtom(atom)
        atom_map[i] = mol_idx  # Store mapping

    # Step 2: Add bonds using adjacency matrix
    num_atoms = len(node_features)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):  # Avoid duplicate bonds
            if adj_matrix[i, j] > 0:
                bond_type = Chem.BondType.SINGLE  # Default to single bond

                # Infer bond order based on known hybridization/aromaticity
                if node_features[i][3] == 2 and node_features[j][3] == 2:  # sp2 hybridization → Double bond
                    if not (node_features[i][0] == 8 or node_features[j][0] == 8):  # Oxygen should not exceed valence 2
                        bond_type = Chem.BondType.DOUBLE
                elif node_features[i][3] == 1 and node_features[j][3] == 1:  # sp hybridization → Triple bond
                    bond_type = Chem.BondType.TRIPLE

                # ✅ Only assign AROMATIC bonds if both atoms are in a ring
                if is_aromatic and in_ring:
                    bond_type = Chem.BondType.AROMATIC

                mol.AddBond(atom_map[i], atom_map[j], bond_type)

    # Step 3: Convert to a standard RDKit molecule and generate 2D coordinates
    mol = mol.GetMol()
    # Chem.SanitizeMol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)

    AllChem.Compute2DCoords(mol)

    img = Draw.MolToImage(mol, size=(300, 300))
    image_list.append(img)

    # Display images
    for i, img in enumerate(image_list):
        plt.figure()
        plt.title(f"Molecule {i+1}")
        plt.imshow(img)
        plt.axis("off")
    plt.show()

def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, indices_file, classes):
    """
    Visualizes molecules with correct bond orders.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Adjacency matrix defining connectivity.
        bond_matrix (scipy.sparse.csr_matrix): Bond order matrix with values 1, 2, 3, or 4.
        feature_matrix (numpy.ndarray): Node feature matrix (first column is atomic numbers).
        indices_file (str): File with node indices for atoms.
        classes (list): Node classes.

    Returns:
        None: Displays molecule images.
    """
    # Load node indices
    node_indices = list(range(8000))  # Adjust based on actual data
    node_to_class = {node: cls for node, cls in zip(node_indices, classes)}

    # Identify connected components (molecules)
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

    images = []
    for k in range(n_components - 2):
        print(f"Processing molecule {k}")

        # Get indices of nodes in this molecule
        component_indices = np.where(labels == k)[0]

        # Extract subgraph
        mol_adj = adj_matrix[component_indices, :][:, component_indices]
        mol_features = feature_matrix[component_indices]
        np.fill_diagonal(mol_adj, 0)

        adjacency_to_molecule(mol_features, mol_adj)

        print(mol_adj)
        print(mol_features)
        # Create RDKit molecule
        mol = Chem.RWMol()
        atom_map = {}  # Maps graph indices to RDKit atom indices

        # Add atoms
        for idx, features in enumerate(mol_features):
            atomic_num = int(features[0])  # Atomic number
            formal_charge = int(features[2])  # Formal charge
            hybridization = int(features[3])  # Hybridization (numeric)
            is_aromatic = bool(features[4])  # **Possible** aromaticity
            in_ring = bool(features[5])  # **Possible** ring membership

            atom = Chem.Atom(atomic_num)
            atom.SetFormalCharge(formal_charge)

            # ✅ Do NOT set IsAromatic until molecule is built (RDKit handles this better)
            mol_idx = mol.AddAtom(atom)
            atom_map[idx] = mol_idx  # Store mapping

        num_atoms = len(mol_features)  # ✅ Corrected num_atoms calculation

        # Step 2: Add bonds using adjacency matrix
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):  # Avoid duplicate bonds
                if mol_adj[i, j] > 0:
                    bond_type = Chem.BondType.SINGLE  # Default to single bond

                    # Infer bond order based on hybridization and aromaticity
                    if mol_features[i][3] == 2 and mol_features[j][
                        3] == 2:  # sp2 hybridization → Possible double bond
                        if not (mol_features[i][0] == 8 or mol_features[j][
                            0] == 8):  # Oxygen (atomic num 8) should not exceed valence 2
                            bond_type = Chem.BondType.DOUBLE
                    if mol_features[i][3] == 1 and mol_features[j][3] == 1:  # sp hybridization → Triple bond
                        bond_type = Chem.BondType.TRIPLE

                    # ✅ Only assign AROMATIC bonds if RDKit later detects a ring
                    if mol_features[i][4] == 1 and mol_features[j][4] == 1:
                        bond_type = Chem.BondType.SINGLE  # Default to SINGLE
                        atom_i = mol.GetAtomWithIdx(atom_map[i])
                        atom_j = mol.GetAtomWithIdx(atom_map[j])

                        # ✅ Only assign AROMATIC bonds if both are actually in a ring
                        if atom_i.IsInRing() and atom_j.IsInRing():
                            bond_type = Chem.BondType.AROMATIC  # ✅ Corrected aromatic bond assignment

                    mol.AddBond(atom_map[i], atom_map[j], bond_type)

        mol = mol.GetMol()  # ✅ Convert to RDKit Mol for processing

        # ✅ Now safely set aromaticity AFTER molecule construction
        for atom in mol.GetAtoms():
            if mol_features[atom.GetIdx()][4] == 1 and atom.IsInRing():  # Only mark aromatic atoms in rings
                atom.SetIsAromatic(True)

        # ✅ Debugging: Check ring membership
        for atom in mol.GetAtoms():
            print(
                f"Atom {atom.GetIdx()}: RDKit Ring={atom.IsInRing()}, FeatureMatrix Ring={mol_features[atom.GetIdx()][5]}")

        print("Checking aromatic atoms before sanitization:")
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and not atom.IsInRing():
                print(f"🚨 ERROR: Atom {atom.GetIdx()} is aromatic but not in a ring! Removing aromatic flag.")
                atom.SetIsAromatic(False)  # ✅ Fix RDKit Kekulization error

        # Sanitize and Draw Molecule
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)

        img = Draw.MolToImage(mol, size=(300, 300))
        images.append(img)


    # Display images
    for i, img in enumerate(images):
        plt.figure()
        plt.title(f"Molecule {i+1}")
        plt.imshow(img)
        plt.axis("off")
    plt.show()

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
    PATH = "/Users/taka/Documents/vqgraph_0204/"
    path = PATH
    EPOCH = 2
    adj_file = f"{path}/sample_adj_{EPOCH}.npz"                     # input data
    feat_file = f"{path}sample_node_feat_{EPOCH}.npz"      # assigned code vector id
    indices_file = f"{path}sample_emb_ind_{EPOCH}.npz"
    bond_file = f"{path}sample_bond_order_{EPOCH}.npz"
    bond_to_edge_file_0 = f"{path}sample_bond_to_edge_{EPOCH}_0.npz"
    bond_to_edge_file_1 = f"{path}sample_bond_to_edge_{EPOCH}_1.npz"

    # arr_bond = getdata(bond_file)   # indices of the input
    # arr_bond_to_edge_0 = getdata(bond_to_edge_file_0)   # indices of the input
    # arr_bond_to_edge_1 = getdata(bond_to_edge_file_1)   # indices of the input
    arr_indices = getdata(indices_file)   # indices of the input
    arr_adj = getdata(adj_file)       # assigned quantized code vec indices
    arr_feat = getdata(feat_file)       # assigned quantized code vec indices
    arr_feat = restore_node_feats(arr_feat)
    node_indices = [int(x) for x in arr_indices.tolist()]
    print(node_indices)

    # Ensure bond_matrix is 2D
    # if len(arr_bond.shape) == 1:
    #     arr_bond = arr_bond.reshape(arr_adj.shape)  # Reshape to adjacency matrix size
    subset_adj_matrix = arr_adj[0:200, 0:200]
    subset_attr_matrix = arr_feat[:200]
    # subset_bond_matrix = arr_bond[:200, :200]

    # -------------------------------------
    # split the matrix into molecules
    # -------------------------------------
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, None, node_indices)


if __name__ == '__main__':
    main()