import matplotlib.pyplot as plt

# path = "/Users/taka/PycharmProjects/VQGraph/Analysis/log_2024_11_11"
path = "/Users/taka/PycharmProjects/VQGraph/Analysis/log_11_12_both_shfld_drp_0.2_codebk_50_8_1_1"
epoch_num = 100


def get_four_lists(kwd):
    with open(path) as f:
        line_list = [x for x in f.readlines() if kwd in x]
    # Nov10 22-46-18: train_known_g, epoch   2, feature_loss:  0.001953| edge_loss:  0.018125| commit_loss:  0.251250, loss_train 2.4667
    feat_loss_list = [float(x.split()[4][:-1]) for x in line_list]
    edge_loss_list = [float(x.split()[6][:-1]) for x in line_list]
    commit_loss_list = [float(x.split()[8][:-1]) for x in line_list]
    # feat_loss_list = [float(x.split()[6][:-1]) for x in line_list]
    # edge_loss_list = [float(x.split()[8][:-1]) for x in line_list]
    # commit_loss_list = [float(x.split()[10][:-1]) for x in line_list]
    model_loss_list = [float(x.split()[-1].strip()) for x in line_list]
    return feat_loss_list, edge_loss_list, commit_loss_list, model_loss_list


def main():
    with open(path) as f:
        train_feat_loss_list, train_edge_loss_list, train_commit_loss_list, train_model_loss_list = get_four_lists("train_known")
        tran_feat_loss_list, tran_edge_loss_list, tran_commit_loss_list, tran_model_loss_list = get_four_lists("test_known")
        ind_feat_loss_list, ind_edge_loss_list, ind_commit_loss_list, ind_model_loss_list = get_four_lists("test_unknown")

        def plot_three(train_list, tran_list, ind_list, plotname):
            plt.figure()
            plt.title(plotname)
            plt.scatter(list(range(epoch_num)), train_list, label='train')
            plt.scatter(list(range(epoch_num)), tran_list, label='tran')
            plt.scatter(list(range(epoch_num)), ind_list, label='ind')
            plt.legend()
            plt.show()

        plot_three(train_feat_loss_list, tran_feat_loss_list, ind_feat_loss_list, "Feature Loss")
        plot_three(train_edge_loss_list, tran_edge_loss_list, ind_edge_loss_list, "Edge Loss")
        plot_three(train_commit_loss_list, tran_commit_loss_list, ind_commit_loss_list, "Commit Loss")
        plot_three(train_model_loss_list, tran_model_loss_list, ind_model_loss_list, "Model Loss")


if __name__ == '__main__':
    main()