
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import h5py

path = "/Users/taka/Downloads/"
namelist = ["codebook.npz", "latent_train_list.npz"]


def density_plot(data):

    # Calculate point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Scatter plot colored by density
    plt.scatter(x, y, c=z, cmap='viridis', s=5)
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot Colored by Density')
    plt.show()


def plot_graph(data):
    # Initialize UMAP with custom parameters
    print(data.shape)

    tsne = TSNE(n_components=2, random_state=44, perplexity=90, n_iter=1000)
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Fit and transform the data
    # embedding = reducer.fit_transform(data)
    # x = embedding[30:, 0]
    # y = embedding[30:, 1]
    # xy = np.vstack([x, y])
    embedding = tsne.fit_transform(data)
    # z = gaussian_kde(xy)(xy)

    plt.figure()
    plt.hist2d(embedding[30:, 0], embedding[30:, 1], bins=50, cmap='viridis')
    plt.colorbar(label='Density')

    # plt.scatter(x, y, c=z, cmap='viridis', s=5)
    # Plot the 2D UMAP projection
    # plt.scatter(embedding[30:, 0], embedding[30:, 1], s=125, c='blue', alpha=0.1)
    plt.scatter(embedding[:30, 0], embedding[:30, 1], s=15, c='red', alpha=0.8)
    plt.show()


def seeinside(filename):
    # filename = "out_emb_list.npz"
    if "codebook" in filename:
        arr = np.load(f"{path}{filename}")["arr_0"]
        arr = np.squeeze(arr)
    else:
        import zipfile
        with zipfile.ZipFile(f"{path}{filename}", 'r') as z:
            z.extractall("extracted_folder")
        print("checking the npz file is done")
        with np.load(f"{path}{filename}", mmap_mode='r') as data:
            with h5py.File("large_file.h5", "w") as h5_file:
                h5_file.create_dataset("arr_0", data=data['arr_0'], chunks=True, compression="gzip")

        with h5py.File("large_file.h5", "r") as h5_file:
            array = h5_file["arr_0"]
            chunk_size = 100000
            for start_idx in range(0, array.shape[0], chunk_size):
                chunk = array[start_idx:start_idx + chunk_size]
                print("Chunk shape:", chunk.shape)
    return arr


def main():
    arr_list = []
    for names in namelist:
        print("################")
        print(names)
        arr = seeinside(names)
        print(arr.shape)
        arr_list.append(arr)
    arr_combined = np.vstack(arr_list)
    print(arr_combined.shape)
    plot_graph(arr_combined)


if __name__ == '__main__':
    main()