
import numpy as np
import umap
import matplotlib.pyplot as plt

path = "/Users/taka/Downloads/"
namelist = ["codebook.npz", "latents_ind.npz"]


def main():
    arr_list = []
    for names in namelist:
        print("################")
        print(names)
        arr = seeinside(names)
        print(arr)
        arr_list.append(arr)
    arr_combined = np.vstack(arr_list)
    print(arr_combined.shape)
    plot_graph(arr_combined)


def seeinside(filename):
    # filename = "out_emb_list.npz"
    name = filename.replace(".npz", "")
    arr = np.load(f"{path}{filename}")["arr_0"]
    arr = np.squeeze(arr)
    # print(arr.shape)
    # print(arr)
    return arr


def plot_graph(data):
    # Initialize UMAP with custom parameters
    print(data[:30])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Fit and transform the data
    embedding = reducer.fit_transform(data)
    print(embedding[:30, :])

    # Plot the 2D UMAP projection
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[30:, 0], embedding[30:, 1], s=5, c='blue', alpha=0.6)
    plt.scatter(embedding[:30, 0], embedding[:30, 1], s=125, c='red', alpha=0.6)
    plt.title("UMAP Projection of High-Dimensional Data")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.show()

if __name__ == '__main__':
    main()