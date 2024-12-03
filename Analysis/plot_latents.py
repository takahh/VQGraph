
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# path = "/Users/taka/Documents/output_20241128/"
MODE = "umap"


def plot_graph(data, mode, epoch):
    # Initialize UMAP or TSNE with custom parameters
    parameter_names = None
    if mode == "tsne":
        perplex = 50
        tsne = TSNE(n_components=2, random_state=44, perplexity=perplex)
        embedding = tsne.fit_transform(data)
        parameter_names = f"tsne: perplex {perplex}"
    elif mode == "umap":
        n_neibogher = 20
        min_dist=0.05
        reducer = umap.UMAP(n_neighbors=n_neibogher, metric='cosine', min_dist=min_dist, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        parameter_names = f"umap: n_neiboughers {n_neibogher}, min_dist {min_dist}"

    plt.figure()
    # Define bin edges to control the size of the bins
    x_range = (-25, 25)  # Range for the x-axis
    y_range = (-25, 25)  # Range for the y-axis
    # x_range = (5, 13)  # Range for the x-axis
    # y_range = (-2, 8)  # Range for the y-axis
    n_bins = 50  # Number of bins for both axes

    plt.hist2d(
        embedding[30:, 0], embedding[30:, 1],
        bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
        cmap='viridis'
    )

    plt.colorbar(label='Density')
    plt.title(parameter_names)
    # Overlay scatter plot
    plt.scatter(embedding[:30, 0], embedding[:30, 1], s=5, c='red', alpha=1)
    plt.savefig(f"./plot_epoch{epoch}")

def getdata(filename):
    # filename = "out_emb_list.npz"
    arr = np.load(f"{filename}")["arr_0"]
    arr = np.squeeze(arr)
    return arr


def main():
    arr_list = []
    for epoch in range(1, 11):
        namelist = [f"/VQGraph/codebook_{epoch}.npz", f"/VQGraph/latent_train_{epoch}.npz"]
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
        plot_graph(arr_combined, MODE, epoch)


if __name__ == '__main__':
    main()