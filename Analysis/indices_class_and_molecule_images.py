import numpy as np
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from rdkit.Geometry import Point2D

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


def visualize_molecules_with_classes_on_atoms(adj_matrix, feature_matrix, indices_file, class_file):
    """
    Visualizes molecules with node classes shown near the atoms.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Combined adjacency matrix for all molecules.
        feature_matrix (numpy.ndarray): Node feature matrix. First column is atomic numbers.
        indices_file (str): Path to file containing node indices for all atoms.
        class_file (str): Path to file containing classes for the nodes.

    Returns:
        None: Displays molecule images with annotated classes near atoms.
    """
    # Step 1: Load indices and classes
    node_indices = indices_file.tolist()
    start_node_index = node_indices[0]
    node_indices = [x - start_node_index for x in node_indices]
    classes = [int(x) for x in class_file.tolist()]
    # Map node indices to classes
    node_to_class = {node: cls for node, cls in zip(node_indices, classes)}

    # Step 2: Identify connected components (molecules)
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    print(n_components)
    # Step 3: Extract and annotate molecules
    images = []
    for i in range(5):
        # Get node indices for this molecule
        component_indices = np.where(labels == i)[0]

        # Extract subgraph
        mol_adj = adj_matrix[component_indices, :][:, component_indices]
        mol_features = feature_matrix[component_indices]

        # Create RDKit molecule
        mol = Chem.RWMol()

        # Add atoms and annotate classes
        atom_labels = {}
        for idx, features in zip(component_indices, mol_features):
            atomic_num = int(features[0])  # First element is the atomic number
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            print("------")
            print(atom_idx)
            print(features)
            # Annotate with superscript class label
            class_label = node_to_class.get(idx, "Unknown")
            if class_label != "Unknown":
                class_label_sup = to_superscript(class_label)
                atom_labels[atom_idx] = f"{Chem.GetPeriodicTable().GetElementSymbol(atomic_num)}{class_label}"
            else:
                atom_labels[atom_idx] = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            print(atom_labels[atom_idx])
                # Annotate with inline format

        # Add bonds
        for x, y in zip(*np.where(mol_adj > 0)):
            if x < y:  # Avoid duplicate bonds
                mol.AddBond(int(x), int(y), Chem.BondType.SINGLE)

        # Sanitize molecule
        Chem.SanitizeMol(mol)
        drawer = Draw.MolDraw2DCairo(CANVAS_WIDTH, CANVAS_HEIGHT)  # Increase the size (width, height)
        options = drawer.drawOptions()
        options.bondLineWidth = 2  # Make bonds thicker if needed
        options.scaleBondWidth = True  # Scale bond width relative to image size
        options.atomLabelFontSize = FONTSIZE
        options.atomLabelPadding = 0.4  # Default is 0.2; increase for more space between labels

        for idx, label in atom_labels.items():
            options.atomLabels[idx] = label  # Assign custom labels to atoms

        # Calculate and set the scale
        minv, maxv = compute_molecule_bounds(mol)

        # Set canvas dimensions
        canvas_width = CANVAS_WIDTH
        canvas_height = CANVAS_HEIGHT

        # Calculate molecule dimensions
        mol_width = maxv.x - minv.x
        mol_height = maxv.y - minv.y

        # Add padding
        padding = 0.1
        mol_width_with_padding = mol_width * (1 + padding)
        mol_height_with_padding = mol_height * (1 + padding)

        # Calculate scale
        scale_x = canvas_width / mol_width_with_padding
        scale_y = canvas_height / mol_height_with_padding
        scale = min(scale_x, scale_y)

        # Center the molecule
        center_x = (canvas_width / scale - mol_width) / 2 - minv.x
        center_y = (canvas_height / scale - mol_height) / 2 - minv.y

        # Center the molecule (convert to integers for SetOffset)
        center_x = int((canvas_width / scale - mol_width) / 2 - minv.x)
        center_y = int((canvas_height / scale - mol_height) / 2 - minv.y)

        drawer.SetOffset(center_x, center_y)

        # Draw the molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Get the binary image data
        img_data = drawer.GetDrawingText()

        # Convert binary image data to an image
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(img_data))
        images.append(img)

    # Step 4: Display images
    for i, img in enumerate(images):
        plt.figure(dpi=150)
        plt.title(f"Molecule {i+1}")
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()  # Automatically adjust spacing
    plt.show()


def main():
    path = "/Users/taka/Documents/VQGraph_out_for_images_0121/"
    input_mol_file = f"{path}/molecules.npz"                     # input data
    class_file = f"{path}embed_ind_indices_first8000_1.npz"      # assigned code vector id
    indices_file = f"{path}idx_test_ind_tosave_first8000_1.npz"  #
    indices_file = f"{path}input_nodes_1.npz"  #

    arr_input = getdata(input_mol_file)   # input molecule graph
    arr_indices = getdata(indices_file)   # indices of the input
    arr_class = getdata(class_file)       # assigned quantized code vec indices
    print(f"node id {arr_indices.shape}, class {arr_class.shape}")
    node_indices = arr_indices.tolist()
    # node_indices = list(range(0, 200))
    # print("node_indices")
    # print(node_indices)

    test_indices = arr_indices[:200]
    # -------------------------------------
    # rebuild attr matrix
    # -------------------------------------
    attr_data = arr_input["attr_data"]
    attr_indices = arr_input["attr_indices"]
    attr_indptr = arr_input["attr_indptr"]
    attr_shape = arr_input["attr_shape"]
    attr_matrix = csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
    ic(node_indices[0])
    subset_attr_matrix = attr_matrix[node_indices[0]:node_indices[0] + 200, :].toarray()
    # subset_attr_matrix = attr_matrix.toarray()

    # -------------------------------------
    # rebuild adj matrix
    # -------------------------------------
    # Assuming you have these arrays from your input
    adj_data = arr_input["adj_data"]
    adj_indices = arr_input["adj_indices"]
    adj_indptr = arr_input["adj_indptr"]
    adj_shape = arr_input["adj_shape"]
    # Reconstruct the sparse adjacency matrix
    adj_matrix = csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
    subset_adj_matrix = adj_matrix[0:200, 0:200].toarray()
    # subset_adj_matrix = adj_matrix.toarray()

    # -------------------------------------
    # split the matrix into molecules
    # -------------------------------------
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, arr_indices, arr_class)


if __name__ == '__main__':
    main()