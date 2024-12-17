
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# path = "/Users/mac/Documents/vq-data/"
path = "/Users/taka/Downloads/"
MODE = "tsne"
# MODE = "umap"


def plot_graph(cb_arr, latent_list, mode, epoch, param, cb_size):
    # Initialize UMAP or TSNE with custom parameters
    parameter_names = None
    embedding = None
    heatmap_colors = ["Blues", "binary", "BuGn"]
    cb_colors = ["blue", "black", "green"]

    if mode == "tsne":
        perplex = param
        n_iter = 5000
        tsne = TSNE(n_components=2, random_state=44, perplexity=perplex, n_iter=n_iter)
        data = np.concatenate((cb_arr, latent_list), axis=0)
        parameter_names = f"umap: perplex {param}, epoch {epoch}, cb {cb_size}"

        # -------------------------------------
        # put all data into tsne
        # -------------------------------------
        embedding = tsne.fit_transform(data)

        # -------------------------------------
        # make two lists
        # -------------------------------------
        cb_arr, latent_list = [], []
        for i in range(3):
            cb_arr.append(embedding[cb_size*i:cb_size*(i+1)])
            latent_list.append(embedding[1000*4:][4000*i:4000*(i+1)])

        # -------------------------------------
        # plot three pairs of data
        # -------------------------------------
        for i in range(3):
            plt.figure()
            # Define bin edges to control the size of the bins
            x_min = min(min(cb_arr[i][:, 0]), min(latent_list[i][:, 0]))
            x_max = max(max(cb_arr[i][:, 0]), max(latent_list[i][:, 0]))
            y_min = min(min(cb_arr[i][:, 1]), min(latent_list[i][:, 1]))
            y_max = max(max(cb_arr[i][:, 1]), max(latent_list[i][:, 1]))
            x_range = (x_min, x_max)  # Range for the x-axis
            y_range = (y_min, y_max)  # Range for the y-axis
            n_bins = 100  # Number of bins for both axes

            # cb_size = 1201

            plt.hist2d(
                embedding[cb_size*4:][:, 0], embedding[cb_size*4:][:, 1],
                bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
                cmap=heatmap_colors[i]
            )
            # Overlay scatter plot
            plt.scatter(cb_arr[i][:, 0], cb_arr[i][:, 1], s=1, c="red", alpha=1)

            # plt.colorbar(label='Density')
            plt.title(f"{parameter_names}, cb {cb_size}")
            # plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=3, c='purple', alpha=1)
            plt.show()
            # plt.savefig(f"./plot_epoch{epoch}")

    # elif mode == "umap":
    #     n_neibogher = param
    #     min_dist = 0.1
    #     n_epochs = 5000
    #     # reducer = umap.UMAP(n_neighbors=n_neibogher, metric='cosine', min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42)
    #     reducer = umap.UMAP(n_neighbors=n_neibogher, min_dist=min_dist, n_epochs=n_epochs, n_components=2, random_state=42).fit(data[cb_size:])
    #     embedding_latent = reducer.transform(data[2*cb_size:])
    #     embedding_quantized = reducer.transform(data[:2*cb_size])
    #     parameter_names = f"umap: n_neiboughers {n_neibogher}, min_dist {min_dist}, epoch {epoch}\n n_epochs {n_epochs}"
    #
    #     plt.figure()
    #     # Define bin edges to control the size of the bins
    #     x_range = (min(embedding_latent[:, 0]), max(embedding_latent[:, 0]))  # Range for the x-axis
    #     y_range = (min(embedding_latent[:, 1]), max(embedding_latent[:, 1]))  # Range for the y-axis
    #     n_bins = 100  # Number of bins for both axes
    #     # cb_size = 1201
    #     plt.hist2d(
    #         embedding_latent[:, 0], embedding_latent[:, 1],
    #         bins=[np.linspace(*x_range, n_bins), np.linspace(*y_range, n_bins)],
    #         cmap='viridis'
    #     )
    #
    #     plt.colorbar(label='Density')
    #     plt.title(f"{parameter_names}, cb {cb_size}")
    #     # Overlay scatter plot
    #     plt.scatter(embedding[cb_size:2 * cb_size, 0], embedding[cb_size:2 * cb_size, 1], s=3, c='red', alpha=1)
    #     # plt.scatter(embedding[:cb_size, 0], embedding[:cb_size, 1], s=3, c='purple', alpha=1)
    #     plt.show()
    #     # plt.savefig(f"./plot_epoch{epoch}")


def getdata(filename):
    # filename = "out_emb_list.npz"
    arr = np.load(f"{filename}")["arr_0"]
    arr = np.squeeze(arr)
    return arr


def main():
    arr_list = []
    DIMENSION = 256
    EPOCH = 17
    for epoch in range(EPOCH, EPOCH + 1):
        arr = None
        print(f"epoch {epoch}")
        namelist = [f"{path}codebook_{epoch}.npz", f"{path}latent_train_{epoch}.npz"]
        # namelist = [f"{path}codebook_{epoch}.npz", f"{path}init_codebook_{epoch}.npz", f"{path}latent_train_{epoch}.npz"]
        for names in namelist:
            arr = getdata(names)
            if "book" in names:
                cb_arr = np.unique(arr, axis=0)[-4:]
                cb_arr = np.reshape(cb_arr, (-1, DIMENSION))
                print()
                cb_size = arr.shape[1]
            else:
                latent_arr = arr
        for param in [5, 10, 100]:
            plot_graph(cb_arr, latent_arr, MODE, epoch, param, cb_size)


if __name__ == '__main__':
    main()