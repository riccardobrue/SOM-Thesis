from _ultimate.som_libs.custom_minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import _ultimate.manage_data.data_normalize as dn
import os
import shutil

# ---------------------------------------
# PARAMETERS
# ---------------------------------------

# checkpoint folder name's format: << "tfckpt_" prefix_1 "_" prefix_2 "_" m "x" n "_" training_over "_ep-" epochs >>
# prefix_1 could be whatever you want
# prefix_2 could be: << "tmp-" epoch >> or << "fin-" epoch >> used to restore a specific checkpoint
# << m "x" n "_" training_over "_ep-" epochs  >> represents the suffix
# suffix indicates the dimension of the SOM and the maximum number of epochs for the relative training
# training_over could be: "net" if the SOM has been trained over network attributes, "hnd" or "fnd" otherwise

folder_prefix_1 = "pc_minisom_"
chart_prefix = "fin-1000_"

epochs = 10000

restore_som = False  # true: doesn't train the som and doesn't store any new checkpoint files
folder_prefix_2 = "tmp-100_minisom_"  # to select the restored checkpoint

checkpoint_iters = 100  # store training som every n iterations

heuristic_size = False  # 22x22 (if false it is needed to specify the "som_side_dim" variable and the "ckpt_folder" name)
manually_picked_som_dim = 60  # if heuristic_size is False, this will be the chosen som's side size

use_reverse = True  # if true: uses the (trained) som over the network attributes instead of the simulation results

use_hnd = True  # false-> uses fnd

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
# LOAD THE NORMALIZED DATA
# ---------------------------------------
nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
    type="equal", shuffle=True)

headers_nt = headers_nt[3:]
nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

# Get the best protocol for each row
sim_hnd_cols = [1, 3, 5, 7]
sim_fnd_cols = [0, 2, 4, 6]

sim_hnd_norm = sim_norm[sim_hnd_cols]
sim_fnd_norm = sim_norm[sim_fnd_cols]

hnd_protocols_names = headers_sim[sim_hnd_cols]
fnd_protocols_names = headers_sim[sim_fnd_cols]

best_hnd_protocols = np.argmax(sim_norm[:, sim_hnd_cols], axis=1)  # index of the most efficient protocol
best_fnd_protocols = np.argmax(sim_norm[:, sim_fnd_cols], axis=1)

print("Data loaded")
print("_______________________________________________________")
print("-- Network attributes: \n", headers_nt, "\n")
print("_______________________________________________________")
print("-- AVG layers: \n", headers_avg_layers, "\n")
print("_______________________________________________________")
print("-- AVG CHxRounds headers: \n", headers_avg_chxrounds, "\n")
print("_______________________________________________________")
print("-- HND protocols names: \n", hnd_protocols_names, "\n")
print("_______________________________________________________")
print("-- FND protocols names: \n", fnd_protocols_names, "\n")
print("_______________________________________________________")
print("-- Simulation data hnd (samples): \n", sim_hnd_norm[:4], "\n")
print("_______________________________________________________")
print("-- Simulation data fnd (samples): \n", sim_fnd_norm[:4], "\n")
print("_______________________________________________________")

# ---------------------------------------
# SELECT THE DATA FOR CLUSTERING
# ---------------------------------------
if use_reverse:
    print("Clustering on network attributes")
    clustering_data = nt_norm
else:
    if use_hnd:
        print("Clustering on simulation results (HND)")
        clustering_data = sim_hnd_norm
    else:
        print("Clustering on simulation results (FND)")
        clustering_data = sim_fnd_norm

print("Clustering data size: ", clustering_data.shape)
print("Clustering data (samples): \n", clustering_data[:4])

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
if use_reverse:
    folder_suffix = folder_suffix + "net_"  # reversed
else:
    if use_hnd:
        folder_suffix = folder_suffix + "hnd_"
    else:
        folder_suffix = folder_suffix + "fnd_"

folder_suffix = folder_suffix + "ep-" + str(epochs)
print("_______________________________________________________")
print("Checkpoint folder suffix: ", folder_suffix)

# ---------------------------------------
# TRAIN, OR RESTORE, THE SOM
# ---------------------------------------
# Initialization and training
som = MiniSom(som_side_dim, som_side_dim, clustering_data.shape[1], sigma=1.0, learning_rate=0.4)

if restore_som:
    pass
if train_som:
    print("_______________________________________________________")
    print("Training the SOM (...)")
    som.random_weights_init(clustering_data)
    # som.train_random(data, 10000)  # random training, pick as starting locations from random input vectors
    som.train_batch(clustering_data, epochs)  # random training
    print("Training done!")

# ---------------------------------------
# GET THE OUTPUT GRID AND COMPUTE U-MATRIX (also called correlation matrix)
# ---------------------------------------
u_matrix = som.distance_map().T

# ---------------------------------------
# CREATE A MAPPED DATA TO KNOW THE BMUs OVER THE DATA
# ---------------------------------------
mapped_data = []
for cnt, xx in enumerate(clustering_data):
    mapped_data.append(som.winner(xx))

mapped_data = np.array(mapped_data)
mapped_data_X = mapped_data[:, 0] + .5
mapped_data_Y = mapped_data[:, 1] + .5

# ======================================================================================================================
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

if use_reverse:
    basic_chart_path = basic_chart_path + "\\_charts\\crt_" + folder_prefix_1 + chart_prefix + folder_suffix + "\\"
else:
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
    if use_reverse:
        plt.title("Best protocols on HND (Training over network attributes) [" + label_names[prot_index] + "]")
    else:
        plt.title("Best protocols on HND (Training over simulation results) [" + label_names[prot_index] + "]")

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
    if use_reverse:
        plt.title("Best protocols on FND (Training over network attributes) [" + label_names[prot_index] + "]")
    else:
        plt.title("Best protocols on FND (Training over simulation results) [" + label_names[prot_index] + "]")

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
    unique_real_classes = np.unique(nt[:, att_index + 3])  # +3 because i remove the first three columns
    print("===")
    print(" - ", unique_classes)
    print(" - ", unique_real_classes)
    # ------------------

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
        if use_reverse:
            plt.title(
                'NT att (Training over network attributes) [' + attribute_name + "] " + str(round(
                    unique_real_classes[att_specific_value_index], 4)))
        else:
            plt.title(
                'NT att (Training over simulation results) [' + attribute_name + "] " + str(round(
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

    print("Protocol: ", label_names[prot_index])
    prot_column_data = sim[:, cols[prot_index]]  # get the results of a specific protocol

    # create different ranges
    max_value = np.amax(prot_column_data)
    min_value = np.amin(prot_column_data)

    ranges = np.linspace(min_value, max_value, num=number_of_ranges + 1).round(0)
    ranges = ranges[1:number_of_ranges + 1]

    print("max: ", max_value, " - min: ", min_value, " - Ranges: ", ranges)
    print(prot_column_data[:6])

    # create the classes (splitted by the range)
    classes = []
    for prot_data in prot_column_data:
        for idx, val in enumerate(ranges):
            if prot_data <= val:
                classes.append(idx)
                break
    unique_classes = np.unique(classes)  # (4,) -> [0 1 2 3]

    print("Classes: ", classes)
    print("Classes size: ", np.array(classes).shape)
    print("Unique classes: ", unique_classes)
    print("\n")

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
        if use_reverse:
            plt.title("Ranges HND (Training over network attributes) [" + label_names[prot_index] + "][<=" + str(ranges[
                                                                                                                     u]) + "]")
        else:
            plt.title("Ranges HND (Training over simulation results) [" + label_names[prot_index] + "][<=" + str(ranges[
                                                                                                                     u]) + "]")

        # SAVE THE CHART
        att_directory_path = directory_path + label_names[prot_index] + "\\"
        if not os.path.exists(att_directory_path):
            os.makedirs(att_directory_path)

        plt.savefig(att_directory_path + 'HND_' + label_names[prot_index] + '_' + str(ranges[u]) + '.png', dpi=my_dpi)

print("Finished!")