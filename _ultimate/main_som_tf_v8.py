import _ultimate.manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import _ultimate.som_libs.SOM_TF_2_ext as som_tf
import os
import shutil
import itertools
import matplotlib.cm as cm

"""
---------------------------
EQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
'HEIGHT'; 'WIDTH'; 'NODE' are removed because calculated inside new column "DENSITY"
'R0' - 0; '%AGGR' - 1; 'HET' - 2; 'HOM ENERGY' - 3; 'HOM RATE' - 4; 'DENSITY' - 5 
---------------------------
EQUAL - PROTOCOLS INDICES:
---------------------------
'REECHD FND' - 0; 'REECHD HND' - 1; 'HEED FND' - 2; 'HEED HND' - 3; 'ERHEED FND' - 4; 'ERHEED HND' - 5; 'FMUC FND' - 6; 'FMUC HND' - 7
---------------------------
UNEQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
---------------------------
UNEQUAL - PROTOCOLS INDICES:
---------------------------
"""


# continue from:
# net only --> continue from 0-5
# all hnd --> continue from 2-5


def execute(clustering_cols, clustering_data_type="all"):
    if len(clustering_cols) == 0:
        att_string = "all_cols"
    else:
        att_string = "-".join(str(x) for x in clustering_cols)

    # ---------------------------------------
    # PARAMETERS ###
    # ---------------------------------------

    # checkpoint folder name's format: << "tfckpt_" prefix_1 "_" prefix_2 "_" m "x" n "_" training_over "_ep-" epochs >>
    # prefix_1 could be whatever you want
    # prefix_2 could be: << "tmp-" epoch >> or << "fin-" epoch >> used to restore a specific checkpoint
    # << m "x" n "_" training_over "_ep-" epochs  >> represents the suffix
    # suffix indicates the dimension of the SOM and the maximum number of epochs for the relative training
    # training_over could be: "net" if the SOM has been trained over network attributes, "hnd" or "fnd" otherwise

    folder_prefix_1 = "pc_" + att_string + "_"
    chart_prefix = "fin-800_"

    epochs = 800

    restore_som = True  # true: doesn't train the som and doesn't store any new checkpoint files
    folder_prefix_2 = "tmp-200_"  # to select the restored checkpoint

    checkpoint_iters = 200  # store training som every n iterations

    heuristic_size = False  # 22x22 (if false it is needed to specify the "som_side_dim" variable and the "ckpt_folder" name)
    manually_picked_som_dim = 36  # if heuristic_size is False, this will be the chosen som's side size

    # charts parameters
    number_of_ranges = 6  # number of ranges which the protocols are splitted in the chart

    # charts storing parameters
    my_dpi = 96
    pixels = 1000
    points_alpha = .35  # chart points transparency (0-1)

    # ---------------------------------------
    # DERIVED PARAMETERS
    # ---------------------------------------

    train_som = not restore_som
    store_som = not restore_som

    # ---------------------------------------
    # CALCULATE LATTICE SIZE
    # ---------------------------------------
    def heuristic_som_size(input_len, size='normal'):
        heuristic_map_units = 5 * input_len ** 0.54321

        if size == 'big':
            heuristic_map_units = 4 * (heuristic_map_units)
        elif size == "small":
            heuristic_map_units = 0.25 * (heuristic_map_units)

        return heuristic_map_units

    # ---------------------------------------
    # COMPUTING THE U-MATRIX
    # ---------------------------------------
    def fast_norm(x):
        """Returns norm-2 of a 1-D numpy array.
        * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
        """
        return np.sqrt(np.dot(x, x.T))

    def distance_map(weights):
        # weights is a 22x22x8 --> som_dim x som_dim x input_features
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = np.zeros((weights.shape[0], weights.shape[1]))  # 22x22 filled with 0
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):  # add 1 column before and 1 after
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):  # add 1 row up and 1 down
                    if 0 <= ii < weights.shape[0] and 0 <= jj < weights.shape[1]:
                        w_1 = weights[ii, jj, :]
                        w_2 = weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1 - w_2)
            it.iternext()
        um = um / um.max()
        return um

    print("=========================================")
    # ---------------------------------------
    # LOAD THE NORMALIZED DATA
    # ---------------------------------------
    nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
        type="equal", shuffle=True)

    headers_nt = headers_nt[3:]
    nt = nt[:, 3:]  # remove the first three columns which are not relevant
    nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

    all_data = np.append(nt, sim, axis=1)
    all_data_norm = np.append(nt_norm, sim_norm, axis=1)

    # Get the best protocol for each row
    sim_hnd_cols = [1, 3, 5, 7]
    sim_fnd_cols = [0, 2, 4, 6]

    sim_hnd_norm = sim_norm[:, sim_hnd_cols]
    sim_fnd_norm = sim_norm[:, sim_fnd_cols]

    all_data_hnd_norm = np.append(nt_norm, sim_hnd_norm, axis=1)
    all_data_fnd_norm = np.append(nt_norm, sim_fnd_norm, axis=1)

    hnd_protocols_names = headers_sim[sim_hnd_cols]
    fnd_protocols_names = headers_sim[sim_fnd_cols]

    best_hnd_protocols = np.argmax(sim_norm[:, sim_hnd_cols], axis=1)  # index of the most efficient protocol
    best_fnd_protocols = np.argmax(sim_norm[:, sim_fnd_cols], axis=1)

    """
    print("Data loaded")
    print("_______________________________________________________")
    print("-- Network attributes: \n", headers_nt, "\n")
    print("_______________________________________________________")
    print("-- AVG layers: \n", headers_avg_layers, "\n")
    print("_______________________________________________________")
    print("-- AVG CHxRounds headers: \n", headers_avg_chxrounds, "\n")
    print("_______________________________________________________")
    print("-- SIM protocols names: \n", headers_sim, "\n")
    print("_______________________________________________________")
    print("-- HND protocols names: \n", hnd_protocols_names, "\n")
    print("_______________________________________________________")
    print("-- FND protocols names: \n", fnd_protocols_names, "\n")
    print("_______________________________________________________")
    print("-- Simulation data hnd (samples): \n", sim_hnd_norm[:4], "\n")
    print("_______________________________________________________")
    print("-- Simulation data fnd (samples): \n", sim_fnd_norm[:4], "")
    print("_______________________________________________________")
    """
    # ---------------------------------------
    # SELECT THE DATA FOR CLUSTERING
    # ---------------------------------------
    print("=========================================================================================== CLUSTERING DATA")
    if clustering_data_type == "all":
        print("Clustering on all the data")
        # clustering_data = clustering_data[np.where(clustering_data[:, 1] == 0.4)]  # take only the rows where %AGGR is 0.4
        clustering_data = all_data_norm
    # -------------------------------------------------------------------------------------------------------------------
    elif clustering_data_type == "net":
        print("Clustering on network attributes (cut)")
        # clustering_data = nt_norm
        # clustering_data = nt_norm[np.where(nt_norm[:, 1] == 0.4)]  # take only the rows where %AGGR is 0.4
        clustering_data = nt_norm[:, clustering_cols]
    # -------------------------------------------------------------------------------------------------------------------
    elif clustering_data_type == "net+hnd":
        print("Clustering on network attributes plus hnd (cut)")
        clustering_data = all_data_hnd_norm[:, clustering_cols]
    elif clustering_data_type == "net+fnd":
        print("Clustering on network attributes plus fnd (cut)")
        clustering_data = all_data_fnd_norm[:, clustering_cols]
    # ------------------------------------------------------------------------------------------------------------------
    elif clustering_data_type == "sim":
        print("Clustering on simulation results (cut)")
        clustering_data = sim_norm[:, clustering_cols]
    elif clustering_data_type == "hnd":
        print("Clustering on simulation results (HND) (cut)")
        clustering_data = sim_hnd_norm[:, clustering_cols]
    elif clustering_data_type == "fnd":
        print("Clustering on simulation results (FND) (cut)")
        clustering_data = sim_fnd_norm[:, clustering_cols]
    # ------------------------------------------------------------------------------------------------------------------
    elif clustering_data_type == "all-sim":
        print("Clustering on all simulation results")
        clustering_data = sim_norm
    elif clustering_data_type == "all-hnd":
        print("Clustering on simulation results (HND)")
        clustering_data = sim_hnd_norm
    elif clustering_data_type == "all-fnd":
        print("Clustering on simulation results (FND)")
        clustering_data = sim_fnd_norm
    # ------------------------------------------------------------------------------------------------------------------
    elif clustering_data_type == "all-net":
        print("Clustering on network attributes ")
        clustering_data = nt_norm

    print("Clustering data size: ", clustering_data.shape)
    print("Clustering data (samples): \n", clustering_data[:4])
    print("===========================================================================================================")

    # ---------------------------------------
    # COMPUTE THE SOM SIZE HEURISTICALLY
    # ---------------------------------------
    if heuristic_size:
        lattice_size = heuristic_som_size(clustering_data.shape[0])  # heuristic lattice size
        som_side_dim = int(lattice_size ** .5)  # compute the lattice width - height size
    else:
        som_side_dim = manually_picked_som_dim

    print("_______________________________________________________")
    print("SOM dimension: ", som_side_dim, "x", som_side_dim)

    # ---------------------------------------
    # CREATE THE CHECKPOINT FOLDER SUFFIX
    # ---------------------------------------
    folder_suffix = str(som_side_dim) + "x" + str(som_side_dim) + "_"
    folder_suffix = folder_suffix + clustering_data_type + "_"

    folder_suffix = folder_suffix + "ep-" + str(epochs)
    print("_______________________________________________________")
    print("Checkpoint folder suffix: ", folder_suffix)
    # ---------------------------------------
    # TRAIN, OR RESTORE, THE SOM
    # ---------------------------------------
    som = som_tf.SOM(m=som_side_dim, n=som_side_dim, dim=clustering_data.shape[1], epochs=epochs,
                     ckpt_prefix_1=folder_prefix_1, ckpt_suffix=folder_suffix)

    if restore_som:
        som.restore(prefix_1=folder_prefix_1, prefix_2=folder_prefix_2, suffix=folder_suffix)
    if train_som:
        print("_______________________________________________________")
        print("Training the SOM (...)")
        # fix or find another system to continue the learning from a certain point (with reduced learning_rate)
        som.train(clustering_data, restart_from=0, checkpoints_iterations=checkpoint_iters)
        # restart_from indicates the iteration number
        # (to decrease the learning rate after a restored session)
        # Stop the training when the QUANTIZATION ERROR falls below a certain threshold (Applications of SOM Johnsson M. - 2.1)

    # ---------------------------------------
    # GET THE OUTPUT GRID AND COMPUTE U-MATRIX (also called correlation matrix)
    # ---------------------------------------
    image_grid = som.get_centroids()
    mat = np.zeros(shape=(som_side_dim, som_side_dim, clustering_data.shape[1]))
    for r in range(0, len(image_grid)):
        for c in range(0, len(image_grid[r])):
            mat[r][c] = image_grid[r][c]

    # np.save('mat',mat)

    u_matrix = distance_map(mat).T

    # ---------------------------------------
    # CREATE A MAPPED DATA TO KNOW THE BMUs OVER THE DATA
    # ---------------------------------------
    mapped_data = np.array(som.map_vects(clustering_data))
    mapped_data_X = mapped_data[:, 0] + .5
    mapped_data_Y = mapped_data[:, 1] + .5

    # ---------------------------------------
    # STORE THE LEARNED SOM AND CLOSE THE TENSORFLOW SESSION
    # ---------------------------------------
    if store_som:
        som.store(prefix_1=folder_prefix_1, prefix_2="fin-" + str(epochs) + "_", suffix=folder_suffix)
    som.close_sess()

    # ======================================================================================================================
    # VISUALIZATION
    # ======================================================================================================================
    # ---------------------------------------
    # VISUALIZING/SAVING THE CHARTS
    # ---------------------------------------
    _figure_counter = 1

    # create basic color palette for the charts
    x = plt.cm.get_cmap('tab10')
    colors = x.colors

    # generate the chart folder
    basic_chart_path = os.path.dirname(os.path.realpath(__file__))

    basic_chart_path = basic_chart_path + "\\_charts\\crt_" + folder_prefix_1 + chart_prefix + folder_suffix + "\\"

    if not os.path.exists(basic_chart_path):
        os.makedirs(basic_chart_path)

    print("_______________________________________________________")
    print("Chart folder's path: ", basic_chart_path)

    # ----------------------------------------------------------------------------------------------------------------------
    # 1st SECTION - Best protocols in HND
    # ----------------------------------------------------------------------------------------------------------------------

    cols = sim_hnd_cols  # (4,) -> [1 3 5 7]
    label_names = hnd_protocols_names  # (4,) -> ['REECHD HND' 'HEED HND' 'ERHEED HND' 'FMUC HND']
    classes = best_hnd_protocols  # (5184,) -> [2 2 2 .... 3 2 1 ... 1 1 2]
    unique_classes = np.unique(classes)  # (4,) -> [0 1 2 3]

    directory_path = basic_chart_path + "\\best_sim_hnd\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for prot_index in range(0, len(cols)):
        plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
        _figure_counter = _figure_counter + 1

        plt.bone()
        plt.pcolor(u_matrix)

        for i, u in enumerate(unique_classes):
            if u == prot_index:
                xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
                yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
                plt.scatter(xi, yi, color=colors[i], label=label_names[u], alpha=points_alpha)

        plt.axis([0, som_side_dim, 0, som_side_dim])
        plt.interactive(True)
        # plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

        # TITLE OF THE CHART
        plt.title(
            "Best protocols on HND (Training over [" + clustering_data_type + "]) [" + label_names[prot_index] + "]")

        # SAVE THE CHART
        plt.savefig(directory_path + 'HND_' + label_names[prot_index] + '.png', dpi=my_dpi)

    # ----------------------------------------------------------------------------------------------------------------------
    # 2nd SECTION - Best protocols in FND
    # ----------------------------------------------------------------------------------------------------------------------

    cols = sim_fnd_cols  # (4,) -> [0 2 4 6]
    label_names = fnd_protocols_names  # (4,) -> ['REECHD FND' 'HEED FND' 'ERHEED FND' 'FMUC FND']
    classes = best_fnd_protocols  # (5184,) -> [1 3 3 .... 1 2 1 ... 2 1 2]
    unique_classes = np.unique(classes)  # (4,) -> [0 1 2 3]

    directory_path = basic_chart_path + "\\best_sim_fnd\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for prot_index in range(0, len(cols)):
        plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
        _figure_counter = _figure_counter + 1

        plt.bone()
        plt.pcolor(u_matrix)

        for i, u in enumerate(unique_classes):
            if u == prot_index:
                xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
                yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
                plt.scatter(xi, yi, color=colors[i], label=label_names[u], alpha=points_alpha)

        plt.axis([0, som_side_dim, 0, som_side_dim])
        plt.interactive(True)
        # plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

        # TITLE OF THE CHART
        plt.title(
            "Best protocols on FND (Training over [" + clustering_data_type + "]) [" + label_names[prot_index] + "]")

        # SAVE THE CHART
        plt.savefig(directory_path + 'FND_' + label_names[prot_index] + '.png', dpi=my_dpi)

    # ----------------------------------------------------------------------------------------------------------------------
    # 3rd SECTION - Network attributes
    # ----------------------------------------------------------------------------------------------------------------------

    attributes = headers_nt  # (6,) -> ['R0' '%AGGR' 'HET' 'HOM ENERGY' 'HOM RATE' 'DENSITY']

    directory_path = basic_chart_path + "\\nt_atts_mapping\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for att_index in range(0, len(attributes)):

        attribute_name = attributes[att_index]
        classes = nt_norm[:, att_index]
        unique_classes = np.unique(classes)

        # ------------------
        # get the real value not the normalized one (this should be mapped with the normalized)
        # ------------------
        unique_real_classes = np.unique(nt[:, att_index])
        """
        print("===")
        print(" - ", unique_classes)
        print(" - ", unique_real_classes)
        # ------------------
        """

        for att_specific_value_index in range(0, len(unique_classes)):
            plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
            _figure_counter = _figure_counter + 1

            plt.bone()
            plt.pcolor(u_matrix)

            for i, u in enumerate(unique_classes):
                if u == unique_classes[att_specific_value_index]:
                    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
                    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
                    plt.scatter(xi, yi, color=colors[i], label=attribute_name + "_" + str(u), alpha=points_alpha)

            plt.axis([0, som_side_dim, 0, som_side_dim])
            plt.interactive(True)
            # plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

            # TITLE OF THE CHART
            plt.title("NT att (Training over [" + clustering_data_type + "]) [" + attribute_name + "] " + str(round(
                unique_real_classes[att_specific_value_index], 4)))

            # SAVE THE CHART
            att_directory_path = directory_path + attribute_name + "\\"
            if not os.path.exists(att_directory_path):
                os.makedirs(att_directory_path)
            plt.savefig(att_directory_path + 'NT_' + attribute_name + "_" + str(att_specific_value_index) + '.png',
                        dpi=my_dpi)

    # ----------------------------------------------------------------------------------------------------------------------
    # 4st SECTION - Ranges in HND
    # ----------------------------------------------------------------------------------------------------------------------

    cols = sim_hnd_cols  # (4,) -> [1 3 5 7]
    label_names = hnd_protocols_names  # (4,) -> ['REECHD HND' 'HEED HND' 'ERHEED HND' 'FMUC HND']
    # classes = best_hnd_protocols  # (5184,) -> [2 2 2 .... 3 2 1 ... 1 1 2]

    directory_path = basic_chart_path + "\\ranges_sim_hnd\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        shutil.rmtree(directory_path, ignore_errors=True)

    for prot_index in range(0, len(cols)):  # iterate each protocol

        # print("Protocol: ", label_names[prot_index])
        prot_column_data = sim[:, cols[prot_index]]  # get the results of a specific protocol

        # create different ranges
        max_value = np.amax(prot_column_data)
        min_value = np.amin(prot_column_data)

        ranges = np.linspace(min_value, max_value, num=number_of_ranges + 1).round(0)
        ranges = ranges[1:number_of_ranges + 1]

        # print("max: ", max_value, " - min: ", min_value, " - Ranges: ", ranges)
        # print(prot_column_data[:6])

        # create the classes (splitted by the range)
        classes = []
        for prot_data in prot_column_data:
            for idx, val in enumerate(ranges):
                if prot_data <= val:
                    classes.append(idx)
                    break
        unique_classes = np.unique(classes)  # (4,) -> [0 1 2 3]
        """
        print("Classes: ", classes)
        print("Classes size: ", np.array(classes).shape)
        print("Unique classes: ", unique_classes)
        print("\n")
        """
        for i, u in enumerate(unique_classes):
            plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
            _figure_counter = _figure_counter + 1

            plt.bone()
            plt.pcolor(u_matrix)

            xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
            yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
            plt.scatter(xi, yi, color=colors[i], label=label_names[prot_index], alpha=points_alpha)

            plt.axis([0, som_side_dim, 0, som_side_dim])
            plt.interactive(True)

            # TITLE OF THE CHART
            plt.title(
                "Ranges HND (Training over [" + clustering_data_type + "]) [" + label_names[prot_index] + "][<=" + str(
                    ranges[u]) + "]")

            # SAVE THE CHART
            att_directory_path = directory_path + label_names[prot_index] + "\\"
            if not os.path.exists(att_directory_path):
                os.makedirs(att_directory_path)

            plt.savefig(att_directory_path + 'HND_' + label_names[prot_index] + '_' + str(ranges[u]) + '.png',
                        dpi=my_dpi)

    # ----------------------------------------------------------------------------------------------------------------------
    # 5rd SECTION - Network attributes merged and showed as background (component planes [chap 1.3 - Self organizing N.N.])
    # (Heatmaps)
    # ----------------------------------------------------------------------------------------------------------------------
    attributes = headers_nt  # (6,) -> ['R0' '%AGGR' 'HET' 'HOM ENERGY' 'HOM RATE' 'DENSITY']

    directory_path = basic_chart_path + "\\nt_component_planes\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for att_index in range(0, len(attributes)):
        attribute_name = attributes[att_index]
        classes = nt_norm[:, att_index]
        unique_classes = np.unique(classes)

        background_matrix = np.zeros((som_side_dim, som_side_dim, 1))

        for i in range(0, len(mapped_data)):
            old_value = background_matrix[mapped_data[i][0]][mapped_data[i][1]][0]
            background_matrix[mapped_data[i][0]][mapped_data[i][1]] = (old_value + classes[i]) / 2
            new_value = background_matrix[mapped_data[i][0]][mapped_data[i][1]][0]

        att_matrix = distance_map(background_matrix).T

        plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
        _figure_counter = _figure_counter + 1

        plt.bone()
        plt.pcolor(att_matrix)

        plt.axis([0, som_side_dim, 0, som_side_dim])
        plt.interactive(True)

        # TITLE OF THE CHART
        plt.title("NT att (Training over [" + clustering_data_type + "]) [" + attribute_name + "] ")

        # SAVE THE CHART
        plt.savefig(directory_path + 'NT_' + attribute_name + '.png', dpi=my_dpi)

    # ----------------------------------------------------------------------------------------------------------------------
    # 6th SECTION - Simulation results merged and showed as background (component planes [chap 1.3 - Self organizing N.N.])
    # (Heatmaps)
    # ----------------------------------------------------------------------------------------------------------------------
    attributes = headers_sim  # (8,) -> ['REECHD FND' 'REECHD HND' 'HEED FND' 'HEED HND' 'ERHEED FND' 'ERHEED HND' 'FMUC FND' 'FMUC HND']

    directory_path = basic_chart_path + "\\sim_component_planes\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for att_index in range(0, len(attributes)):
        prot_name = attributes[att_index]
        values = sim_norm[:, att_index]

        background_matrix = np.zeros((som_side_dim, som_side_dim, 1))

        for i in range(0, len(mapped_data)):
            old_value = background_matrix[mapped_data[i][0]][mapped_data[i][1]][0]
            background_matrix[mapped_data[i][0]][mapped_data[i][1]] = (old_value + values[i]) / 2
            new_value = background_matrix[mapped_data[i][0]][mapped_data[i][1]][0]

        att_matrix = distance_map(background_matrix).T

        plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
        _figure_counter = _figure_counter + 1

        plt.bone()
        plt.pcolor(att_matrix)

        plt.axis([0, som_side_dim, 0, som_side_dim])
        plt.interactive(True)

        # TITLE OF THE CHART
        plt.title("SIM res (Training over [" + clustering_data_type + "]) [" + prot_name + "] ")

        # SAVE THE CHART
        plt.savefig(directory_path + 'PROT_' + prot_name + '.png', dpi=my_dpi)
    """
    # ----------------------------------------------------------------------------------------------------------------------
    # 6rd SECTION - Network attributes, visualization through code/weights vectors (polar plot)
    # ----------------------------------------------------------------------------------------------------------------------
    
    attributes = headers_nt  # (6,) -> ['R0' '%AGGR' 'HET' 'HOM ENERGY' 'HOM RATE' 'DENSITY']
    
    directory_path = basic_chart_path + "\\nt_atts_codes\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print("-------------")
    print(mat[:4])
    print(mat.shape)
    
    main_fig = plt.figure(_figure_counter, figsize=(pixels / my_dpi, pixels / my_dpi), dpi=my_dpi)
    _figure_counter = _figure_counter + 1
    
    columns = som_side_dim
    rows = som_side_dim
    
    i = 0
    
    for r in range(0, mat.shape[0]):
        for c in range(0, mat.shape[1]):
            i = i + 1
            val_array = mat[r][c]
    
            sub_fig = plt.figure(figsize=(8, 8))
            ax = sub_fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    
            N = mat.shape[2]
            theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
            radii = 10 * np.random.rand(N)
            width = np.pi / 4 * np.random.rand(N)
            bars = ax.bar(theta, radii, width=width, bottom=0.0)
            for r, bar in zip(radii, bars):
                bar.set_facecolor(cm.jet(r / 10.))
                bar.set_alpha(0.5)
    
            main_fig.add_subplot(rows, columns, i)
            plt.imshow(sub_fig)
    
    plt.show()
    """

    print("Finished!")


if __name__ == "__main__":

    net_attributes = [0, 1, 2, 3, 4, 5]  # ['R0' '%AGGR' 'HET' 'HOM ENERGY' 'HOM RATE' 'DENSITY']
    sim_results = [0, 1, 2, 3, 4, 5, 6,
                   7]  # ['REECHD FND' 'REECHD HND' 'HEED FND' 'HEED HND' 'ERHEED FND' 'ERHEED HND' 'FMUC FND' 'FMUC HND']
    sim_h_f_nd_results = [0, 1, 2, 3]  # ['REECHD H/FND' 'HEED H/FND' 'ERHEED H/FND' 'FMUC H/FND']

    # change this!
    clust_type = "all-fnd"  # clustering type: [all, net, net+hnd, net+fnd, sim, hnd, fnd, all-net, all-sim, all-hnd, all-fnd]

    # ------------------------------------------------------------------
    if clust_type == "net":
        arr = net_attributes
    elif clust_type == "net+hnd" or clust_type == "net+fnd":
        arr = net_attributes  # will be added the hnd/fnd part later
    elif clust_type == "sim":
        arr = sim_results
    elif clust_type == "hnd" or clust_type == "fnd":
        arr = sim_h_f_nd_results
    else:  # all, all-net, all-sim, all-hnd, all-fnd
        arr = []

    # ------------------------------------------------------------------
    if len(arr) == 0:
        execute(clustering_cols=[], clustering_data_type=clust_type)
    else:
        counter = 0
        for i in range(0, len(arr)):
            combs = list(itertools.combinations(arr, i + 1))
            for c in combs:
                columns = np.array(c)
                if clust_type == "net+hnd" or clust_type == "net+fnd":
                    columns = np.concatenate((columns, [6, 7, 8, 9]))  # indeces of hnd/fnd cols in all_data_h/fnd_norm

                print("---> Selected columns: ", columns)
                execute(clustering_cols=columns, clustering_data_type=clust_type)
                counter = counter + 1
        print("Total iterations: ", counter)
