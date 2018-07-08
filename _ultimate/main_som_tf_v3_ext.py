import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import som_libs.SOM_TF_2_ext as som_tf
import os

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

# ---------------------------------------
# PARAMETERS
# ---------------------------------------
# ----------------
# training-restoring parameters
# ----------------

folder_prefix = "ok_ext_"
epochs = 5000

restore_som = True  # true: doesn't train the som and doesn't store any new checkpoint files

heuristic_size = True  # 22x22 (if false it is needed to specify the "som_side_dim" variable and the "ckpt_folder" name)
manually_picked_som_dim = 30  # if heuristic_size is False, this will be the chosen som's side size

use_reverse = True  # if true: uses the (trained) som over the network attributes instead of the simulation results

use_hnd = False  # false-> uses fnd

# ----------------
# Visualization parameters
# ----------------
att_index = 1  # network attribute to be visualized over the som's chart

""" see the values inside @clustering_data """
show_net_att_n = 0.  # specify which attribute to see
show_prot_n = 3  # specify which best protocol to see

# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------
if heuristic_size:
    ckpt_folder_size_name = "22x22"
else:
    ckpt_folder_size_name = str(manually_picked_som_dim) + "x" + str(manually_picked_som_dim)

if use_reverse:
    ckpt_folder = folder_prefix + ckpt_folder_size_name + "_rev_"  # reversed
else:
    if use_hnd:
        ckpt_folder = folder_prefix + ckpt_folder_size_name + "_hnd_"
    else:
        ckpt_folder = folder_prefix + ckpt_folder_size_name + "_fnd_"

ckpt_folder = ckpt_folder + str(epochs)  # the folder name is composed by "cpkt_folder" string + epochs number

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
nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
    type="equal")

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
print("network attributes: \n", headers_nt)
print("avg layers: \n", headers_avg_layers)
print("avg CHxRounds headers: \n", headers_avg_chxrounds)
print("hnd protocols names: \n", hnd_protocols_names)
print("fnd protocols names: \n", fnd_protocols_names)
print("Simulation data hnd (samples): \n", sim_hnd_norm[:4])
print("Simulation data fnd (samples): \n", sim_fnd_norm[:4])

# ---------------------------------------
# SELECT THE DATA FOR CLUSTERING
# ---------------------------------------
if use_reverse:
    clustering_data = nt_norm
else:
    if use_hnd:
        clustering_data = sim_hnd_norm
    else:
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

print("SOM dimension: ", som_side_dim, "x", som_side_dim)

# ---------------------------------------
# TRAIN, OR RESTORE, THE SOM
# ---------------------------------------
som = som_tf.SOM(som_side_dim, som_side_dim, clustering_data.shape[1], epochs=epochs, ckpt_folder_name=ckpt_folder)

if restore_som:
    som.restore()
if train_som:
    print("Training the SOM:")
    # fix or find another system to continue the learning from a certain point (with reduced learning_rate)
    som.train(clustering_data, restart_from=0)
    # restart_from indicates the iteration number
    # (to decrease the learning rate after a restored session)

# ---------------------------------------
# GET THE OUTPUT GRID AND COMPUTE U-MATRIX
# ---------------------------------------
image_grid = som.get_centroids()
mat = np.zeros(shape=(som_side_dim, som_side_dim, clustering_data.shape[1]))
for r in range(0, len(image_grid)):
    for c in range(0, len(image_grid[r])):
        mat[r][c] = image_grid[r][c]
u_matrix = distance_map(mat)

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
    som.store()
som.close_sess()

# ---------------------------------------
# VISUALIZING THE CHARTS
# ---------------------------------------
"""
if use_hnd:
    protocols_names = hnd_protocols_names
    best_protocols = best_hnd_protocols
else:
    protocols_names = fnd_protocols_names
    best_protocols = best_fnd_protocols
"""
figure_counter = 1

# create basic color palette for the charts
x = plt.cm.get_cmap('tab10')
colors = x.colors

# get the default directory for the resulting charts
basic_chart_path = os.path.dirname(os.path.realpath(__file__))

if use_reverse:
    basic_chart_path = basic_chart_path + "\\_charts_" + str(epochs) + "_ep_nt\\"
else:
    basic_chart_path = basic_chart_path + "\\_charts_" + str(epochs) + "_ep_sim\\"

print("Chart path: ", basic_chart_path)
if not os.path.exists(basic_chart_path):
    os.makedirs(basic_chart_path)

# ----------------------------------------------------------------------------------------------------------------------
# 1st SECTION - Best protocols in HND
# ----------------------------------------------------------------------------------------------------------------------

cols = sim_hnd_cols
label_names = hnd_protocols_names
classes = best_hnd_protocols

unique_classes = np.unique(classes)
for prot_index in range(0, len(cols)):
    plt.figure(figure_counter)
    figure_counter = figure_counter + 1
    plt.bone()
    plt.pcolor(u_matrix.T)

    for i, u in enumerate(unique_classes):
        if u == prot_index:
            xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
            yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
            plt.scatter(xi, yi, color=colors[i], label=label_names[u], alpha=.15)

    plt.axis([0, som_side_dim, 0, som_side_dim])
    plt.interactive(True)
    plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

    if use_reverse:
        plt.title(
            'Best protocols on HND (Training over network)' + str(prot_index) + "_" + label_names[prot_index])
    else:
        plt.title(
            'Best protocols on HND (Training over sim_data)' + str(prot_index) + "_" + label_names[prot_index])

    directory_path = basic_chart_path + "\\sim_hnd\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.savefig(directory_path + 'HND_' + str(prot_index) + '_' + label_names[prot_index] + '.png')
    # plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 2nd SECTION - Best protocols in FND
# ----------------------------------------------------------------------------------------------------------------------

cols = sim_fnd_cols
label_names = fnd_protocols_names
classes = best_fnd_protocols
unique_classes = np.unique(classes)

for prot_index in range(0, len(cols)):
    plt.figure(figure_counter)
    figure_counter = figure_counter + 1
    plt.bone()
    plt.pcolor(u_matrix.T)

    for i, u in enumerate(unique_classes):
        if u == prot_index:
            xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
            yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
            plt.scatter(xi, yi, color=colors[i], label=label_names[u], alpha=.15)

    plt.axis([0, som_side_dim, 0, som_side_dim])
    plt.interactive(True)
    plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

    if use_reverse:
        plt.title(
            'Best protocols on FND (Training over network)' + str(prot_index) + "_" + label_names[prot_index])
    else:
        plt.title(
            'Best protocols on FND (Training over sim_data)' + str(prot_index) + "_" + label_names[prot_index])

    directory_path = basic_chart_path + "\\sim_fnd\\"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    plt.savefig(directory_path + 'FND_' + str(prot_index) + '_' + label_names[prot_index] + '.png')
    # plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 3rd SECTION - Network attributes
# ----------------------------------------------------------------------------------------------------------------------

for att_index in range(0, len(headers_nt)):

    attribute_name = headers_nt[att_index]
    classes = nt_norm[:, att_index]
    unique_classes = np.unique(classes)

    for att_specific_value_index in range(0, len(unique_classes)):

        plt.figure(figure_counter)
        figure_counter = figure_counter + 1
        plt.bone()
        plt.pcolor(u_matrix.T)

        for i, u in enumerate(unique_classes):
            if u == unique_classes[att_specific_value_index]:
                xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
                yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
                plt.scatter(xi, yi, color=colors[i], label=attribute_name + "_" + str(u), alpha=.15)

        plt.axis([0, som_side_dim, 0, som_side_dim])
        plt.interactive(True)
        plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))

        if use_reverse:
            plt.title(
                'NT att (Training over network)' + str(att_index) + "_" + attribute_name + "_" + str(
                    att_specific_value_index))
        else:
            plt.title(
                'NT att (Training over sim_data)' + str(att_index) + "_" + attribute_name + "_" + str(
                    att_specific_value_index))

        directory_path = basic_chart_path + "\\nt\\" + attribute_name + "\\"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        plt.savefig(directory_path + 'NT_' + str(att_index) + '_' + attribute_name + "_" + str(
            att_specific_value_index) + '.png')
