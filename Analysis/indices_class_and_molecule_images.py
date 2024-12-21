import numpy as np
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from scipy.sparse.csgraph import connected_components
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


def getdata(filename):
    # filename = "out_emb_list.npz"
    if "mol" in filename:
        arr = np.load(f"{filename}")
    else:
        arr = np.load(f"{filename}")["arr_0"]
    # arr = np.squeeze(arr)
    return arr


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
    classes = class_file.tolist()
    print("node_indices")
    print(node_indices)
    print("classes")
    print(classes)

    # Map node indices to classes
    node_to_class = {node: cls for node, cls in zip(node_indices, classes)}

    # Step 2: Identify connected components (molecules)
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

    # Step 3: Extract and annotate molecules
    images = []
    for i in range(n_components):
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

            # Annotate the atom label with its class
            class_label = node_to_class.get(idx, "Unknown")
            atom_labels[atom_idx] = f"{Chem.GetPeriodicTable().GetElementSymbol(atomic_num)}({class_label})"

        # Add bonds
        for x, y in zip(*np.where(mol_adj > 0)):
            if x < y:  # Avoid duplicate bonds
                mol.AddBond(int(x), int(y), Chem.BondType.SINGLE)

        # Sanitize molecule
        Chem.SanitizeMol(mol)

        # Draw molecule with atom labels
        drawer = Draw.MolDraw2DCairo(400, 400)  # Set the size of the image
        options = drawer.drawOptions()
        for idx, label in atom_labels.items():
            options.atomLabels[idx] = label  # Assign custom labels to atoms

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
        plt.figure()
        plt.title(f"Molecule {i+1}")
        plt.imshow(img)
        plt.axis("off")
    # plt.show()


def main():
    path = "/Users/taka/Documents/output_of_vqgraph_for_analysis/"
    input_mol_file = f"{path}/molecules.npz"                     # input data
    class_file = f"{path}embed_ind_indices_first8000_1.npz"      # assigned code vector id
    indices_file = f"{path}idx_test_ind_tosave_first8000_1.npz"  #

    arr_input = getdata(input_mol_file)   # input molecule graph
    arr_indices = getdata(indices_file)   # indices of the input
    arr_class = getdata(class_file)       # assigned quantized code vec indices

    test_indices = arr_indices[:200]
    # -------------------------------------
    # rebuild attr matrix
    # -------------------------------------
    attr_data = arr_input["attr_data"]
    attr_indices = arr_input["attr_indices"]
    attr_indptr = arr_input["attr_indptr"]
    attr_shape = arr_input["attr_shape"]
    attr_matrix = csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
    subset_attr_matrix = attr_matrix[:3000, :].toarray()

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
    subset_adj_matrix = adj_matrix[:3000, :3000].toarray()

    # -------------------------------------
    # split the matrix into molecules
    # -------------------------------------
    visualize_molecules_with_classes_on_atoms(subset_adj_matrix, subset_attr_matrix, arr_indices, arr_class)


if __name__ == '__main__':
    main()