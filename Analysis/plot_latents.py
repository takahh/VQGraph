
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# import h5py
MODE = "tsne"
# path = "/Users/taka/Documents/output_20241128/"
path = "/VQGraph/outputs/inductive/split_rate_0.2/molecules/SAGE/seed_0/"
namelist = ["codebook.npz", "latent_train_list.npz"]


def plot_graph(data, mode):
    # Initialize UMAP or TSNE with custom parameters
    if mode == "tsne":
        tsne = TSNE(n_components=2, random_state=44, perplexity=50, n_iter=250)
        embedding = tsne.fit_transform(data)
    elif mode == "umap":
        reducer = umap.UMAP(n_neighbors=80, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)

    # Compute the range of the data
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    x_min, x_max = -5, 5
    y_min, y_max =  -5, 5

    # Add margins with zero density
    padding_factor = 0.1  # 10% padding
    x_padding = (x_max - x_min) * padding_factor
    y_padding = (y_max - y_min) * padding_factor

    # Define new ranges for the plot
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    plt.figure()
    # Define bins explicitly to include the padded range
    plt.hist2d(
        embedding[30:, 0], embedding[30:, 1],
        bins=[np.linspace(*x_range, 50), np.linspace(*y_range, 50)],
        cmap='viridis'
    )
    plt.colorbar(label='Density')

    # Overlay scatter plot
    plt.scatter(embedding[:30, 0], embedding[:30, 1], s=5, c='red', alpha=1)

    plt.xlim(x_range)
    plt.ylim(y_range)

    plt.show()

def getdata(filename):
    # filename = "out_emb_list.npz"
    arr = np.load(f"{path}{filename}")["arr_0"]
    arr = np.squeeze(arr)
    return arr


def main():
    arr_list = []
    for names in namelist:
        print("################")
        print(names)
        arr = getdata(names)
        if "book" in names:
            arr = np.unique(arr, axis=0)
            print("book")
        print(arr.shape)
        arr_list.append(arr)
    arr_combined = np.vstack(arr_list)
    print(arr_combined.shape)
    plot_graph(arr_combined, MODE)


if __name__ == '__main__':
    main()