path = "/Users/taka/Downloads/outputs 2/inductive/split_rate_0.2/molecules/SAGE/seed_0/"

namelist = ["codebook_embeddings.npz", "out_emb_list.npz", "tea_soft_labels.npz", "tea_soft_token_assignments.npz"]
# log
# loss_and_score.npz

import numpy as np


def main():
    for names in namelist:
        print("################")
        print(names)
        seeinside(names)


def seeinside(filename):
    filename = "out_emb_list.npz"
    arr = np.load(f"{path}{filename}")
    for files in arr.files:
        print(files)
        print(arr[files].shape)
        print(arr[files])


if __name__ == '__main__':
    main()