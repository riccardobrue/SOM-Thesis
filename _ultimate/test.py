from _ultimate.som_libs.custom_minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import _ultimate.manage_data.data_normalize as dn
import os


def execute(clustering_cols):
    att_string = "-".join(str(x) for x in clustering_cols)

    folder_prefix_1 = "pc_x_ms_" + att_string + "_"
    chart_prefix = "fin-1000_"

    epochs = 100
    manually_picked_som_dim = 30  # if heuristic_size is False, this will be the chosen som's side size


    # charts storing parameters
    my_dpi = 96
    pixels = 1000
    points_alpha = .35  # chart points transparency (0-1)

    # ---------------------------------------
    # LOAD THE NORMALIZED DATA
    # ---------------------------------------
    nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
        type="equal", shuffle=True)

    headers_nt = headers_nt[3:]
    nt = nt[:, 3:]  # remove the first three columns which are not relevant
    nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

    all_data_norm = np.append(nt_norm, sim_norm, axis=1)
    clustering_data = all_data_norm[:, clustering_cols]

    print(clustering_data[:8])

    som_side_dim = manually_picked_som_dim

    folder_suffix = str(som_side_dim) + "x" + str(som_side_dim) + "_"
    folder_suffix = folder_suffix + "all_"

    folder_suffix = folder_suffix + "ep-" + str(epochs)

    som = MiniSom(som_side_dim, som_side_dim, clustering_data.shape[1], sigma=1.0, learning_rate=0.4)

    som.random_weights_init(clustering_data)
    # som.train_random(data, 10000)  # random training, pick as starting locations from random input vectors
    som.train_batch(clustering_data, epochs)

    u_matrix = som.distance_map().T

    mapped_data = []
    for cnt, xx in enumerate(clustering_data):
        mapped_data.append(som.winner(xx))

    mapped_data = np.array(mapped_data)
    mapped_data_X = mapped_data[:, 0] + .5
    mapped_data_Y = mapped_data[:, 1] + .5

    # ======================================================================================================================
    # VISUALIZATION
    # ======================================================================================================================
    _figure_counter = 1

    # create basic color palette for the charts
    x = plt.cm.get_cmap('tab10')
    colors = x.colors

    # generate the chart folder
    basic_chart_path = os.path.dirname(os.path.realpath(__file__))

    basic_chart_path = basic_chart_path + "\\_charts\\crt_" + folder_prefix_1 + chart_prefix + folder_suffix + "\\"

    if not os.path.exists(basic_chart_path):
        os.makedirs(basic_chart_path)

    print("Chart folder's path: ", basic_chart_path)

    # ------------------------------------------------------------------------------------------------------------------
    # 1st SECTION - Best protocols in HND
    # ------------------------------------------------------------------------------------------------------------------

    directory_path = basic_chart_path + "\\u_matrix\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
    _figure_counter = _figure_counter + 1

    plt.bone()
    plt.pcolor(u_matrix)

    plt.axis([0, som_side_dim, 0, som_side_dim])
    plt.interactive(True)
    # plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

    # TITLE OF THE CHART
    plt.title("U-MATRIX")

    # SAVE THE CHART
    plt.savefig(directory_path + 'U-MATRIX_'+att_string+'_.png', dpi=my_dpi)






if __name__ == "__main__":
    net_att_indeces = [0, 1, 2, 3, 4, 5]  # ['R0' '%AGGR' 'HET' 'HOM ENERGY' 'HOM RATE' 'DENSITY']
    sim_indeces = [6, 7, 8, 9, 10, 11, 12, 13]  # for "all" continuing from net_attributes

    # clustering one net att with one result
    for nt_index in net_att_indeces:
        for sim_index in sim_indeces:
            print("---------> ", nt_index, " - ", sim_index)
            execute(clustering_cols=[nt_index, sim_index])
