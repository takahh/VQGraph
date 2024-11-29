
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import h5py
MODE = "umap"
path = "/Users/taka/Documents/output_20241128/"
namelist = ["codebook.npz", "latent_train_list.npz"]


def plot_graph(data, mode):
    # Initialize UMAP with custom parameters
    if mode == "tsne":
        tsne = TSNE(n_components=2, random_state=44, perplexity=130, n_iter=250)
        embedding = tsne.fit_transform(data)
    elif mode == "umap":
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)

    plt.figure()
    plt.hist2d(embedding[30:, 0], embedding[30:, 1], bins=50, cmap='viridis')
    plt.colorbar(label='Density')

    plt.scatter(embedding[:30, 0], embedding[:30, 1], s=15, c='red', alpha=0.8)
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
        print(arr.shape)
        arr_list.append(arr)
    arr_combined = np.vstack(arr_list)
    print(arr_combined.shape)
    plot_graph(arr_combined, MODE)


if __name__ == '__main__':
    main()