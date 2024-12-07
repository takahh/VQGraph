
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

path = "/Users/taka/Documents/output_20241128/"
MODE = "tsne"


def plot_graph(data, mode, epoch, param, cb_size):
    # Initialize UMAP or TSNE with custom parameters
    parameter_names = None
    if mode == "tsne":
        perplex = param
        n_iter = 5000
        tsne = TSNE(n_components=2, random_state=44, perplexity=perplex, n_iter=n_iter)
        embedding = tsne.fit_transform(data)
        parameter_names = f"tsne: perplex {perplex}, epoch {epoch}, n_iter {n_iter}"
    elif mode == "umap":
        n_neibogher = param
        min_dist = 0.1
        n_epochs = 5000
        # reducer = umap.UMAP(n_neighbors=n_neibogher, metric='cosine', min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42)
        reducer = umap.UMAP(n_neighbors=n_neibogher, min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        parameter_names = f"umap: n_neiboughers {n_neibogher}, min_dist {min_dist}, epoch {epoch}\n n_epochs {n_epochs}"

    plt.figure()
    limit_value = 100
    # Define bin edges to control the size of the bins
    x_range = (-10, 30)  # Range for the x-axis
    y_range = (0, 30)  # Range for the y-axis
    x_range = (-limit_value, limit_value)  # Range for the x-axis
    y_range = (-limit_value, limit_value)  # Range for the y-axis
    n_bins = 100  # Number of bins for both axes
    # cb_size = 1201
    plt.hist2d(
        embedding[cb_size:, 0], embedding[cb_size:, 1],
        bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
        cmap='viridis'
    )

    plt.colorbar(label='Density')
    plt.title(f"{parameter_names}, cb {cb_size}")
    # Overlay scatter plot
    plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=5, c='red', alpha=1)
    plt.show()
    # plt.savefig(f"./plot_epoch{epoch}")

def getdata(filename):
    # filename = "out_emb_list.npz"
    arr = np.load(f"{filename}")["arr_0"]
    arr = np.squeeze(arr)
    return arr


def main():
    print(f"plot start...")
    arr_list = []
    for epoch in range(1, 50):
        if epoch != 41:
            continue
        print(f"epoch {epoch}")
        namelist = [f"{path}codebook_{epoch}.npz", f"{path}latent_train_{epoch}.npz"]
        for names in namelist:
            print("################")
            print(names)
            arr = getdata(names)
            if "book" in names:
                arr = np.unique(arr, axis=0)
                cb_size = arr.shape[0]
                print("book")
            else:
                random_indices = np.random.choice(arr.shape[0], 10000, replace=True)
                arr = arr[random_indices]
            print(arr.shape)
            arr_list.append(arr)
        arr_combined = np.vstack(arr_list)
        print(arr_combined.shape)
        # for param in [5, 10, 20, 30, 40, 50]:
        for param in [110]:
            plot_graph(arr_combined, MODE, epoch, param, cb_size)


if __name__ == '__main__':
    main()