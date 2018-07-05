import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import som_libs.SOM_TF_2_ext as som_tf

"""
---------------------------
EQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
'HEIGHT' - 0; 'WIDTH' - 1; 'NODE' - 2; 'R0' - 3; '%AGGR' - 4; 'HET' - 5; 'HOM ENERGY' - 6; 'HOM RATE' - 7; 
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
epochs = 100

restore_som = True  # true: doesn't train the som and doesn't store any new checkpoint files

heuristic_size = True  # 22x22 (if false it is needed to specify the "som_side_dim" variable and the "ckpt_folder" name)
manually_picked_som_dim = 30  # if heuristic_size is False, this will be the chosen som's side size

use_hnd = False  # false-> uses fnd

use_reverse = False  # if true: uses the (trained) som over the network attributes instead of the simulation results

# ----------------
# Visualization parameters
# ----------------
att_index = 6  # network attribute to be visualized over the som's chart

# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------
if heuristic_size:
    ckpt_folder_size_name = "22x22"
else:
    ckpt_folder_size_name = str(manually_picked_som_dim) + "x" + str(manually_picked_som_dim)

if use_hnd:
    if use_reverse:
        ckpt_folder = "ok_" + ckpt_folder_size_name + "_hnd_rev_"  # reversed
    else:
        ckpt_folder = "ok_" + ckpt_folder_size_name + "_hnd_"
else:
    if use_reverse:
        ckpt_folder = "ok_" + ckpt_folder_size_name + "_fnd_rev_"  # reversed
    else:
        ckpt_folder = "ok_" + ckpt_folder_size_name + "_fnd_"

train_som = not restore_som
store_som = not restore_som
ckpt_folder = ckpt_folder + str(epochs)  # the folder name is composed by "cpkt_folder" string + epochs number


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
    """Returns the distance map of the weights.
    Each cell is the normalised sum of the distances between
    a neuron and its neighbours."""
    um = np.zeros((weights.shape[0], weights.shape[1]))
    it = np.nditer(um, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
            for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
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
all_data_equal, net_topology_att_data_equal, sim_data_equal, nt_headers_equal, sim_headers_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal, nt_headers_unequal, sim_headers_unequal = dn.load_normalized_unequal_data()
# Get the best protocol for each row
if use_hnd:
    sim_headers = [1, 3, 5, 7]
else:
    sim_headers = [0, 2, 4, 6]

best_protocols = np.argmax(sim_data_equal[:, sim_headers],
                           axis=1)  # returns the index of the most efficient protocol
best_protocols_names = sim_headers_equal[sim_headers]

print("Data loaded: ")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)

# ---------------------------------------
# SELECT THE DATA FOR CLUSTERING
# ---------------------------------------
if use_reverse:
    clustering_data = net_topology_att_data_equal
else:
    clustering_data = sim_data_equal

print("Clustering data size: ", clustering_data.shape)

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
# TRAIN THE SOM
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
# ----------------------------------------------------------------------------------------------------------------------
# FIGURE 1
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(1)
plt.bone()  # grayscale colors
# @todo: Which should i take? the transposed or the non-transposed one?
# plt.pcolor(u_matrix)  # plotting the U-MATRIX as background
plt.pcolor(u_matrix.T)  # plotting the transposed U-MATRIX as background (?)
print("Protocols names: ", best_protocols_names)
classes = best_protocols
unique_classes = np.unique(classes)

# markers = ['*', 'o', 'D', 'x', 's', 'd', '.', '+']
# create one color and one mark for each class
x = plt.cm.get_cmap('tab10')
colors = x.colors

for i, u in enumerate(unique_classes):
    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
    plt.scatter(xi, yi, color=colors[i], label=best_protocols_names[u], alpha=.5)

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(True)
plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))
if use_hnd:
    if use_reverse:
        plt.title('Best protocols on HND (Training over network)')
    else:
        plt.title('Best protocols on HND (Training over sim_data)')
else:
    if use_reverse:
        plt.title('Best protocols on FND (Training over network)')
    else:
        plt.title('Best protocols on FND (Training over sim_data)')

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# FIGURE 2
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(2)
plt.bone()  # grayscale colors
plt.pcolor(u_matrix.T)  # plotting the U-MATRIX as background

header = nt_headers_equal[att_index]

classes = net_topology_att_data_equal[:, att_index]
unique_classes = np.unique(classes)

print("Distinct values: ", unique_classes)

# create one color and one mark for each class
x = plt.cm.get_cmap('tab10')
colors = x.colors

for i, u in enumerate(unique_classes):
    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
    plt.scatter(xi, yi, color=colors[i], label=header + " " + str(round(u, 2)), alpha=.5)

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(False)
plt.legend(loc='center left', bbox_to_anchor=(0, 1.08))
plt.title('Visualizing (normalized) ' + header)

if use_reverse:
    plt.title('Visualizing (normalized) ' + header + " (Training over network)")
else:
    plt.title('Visualizing (normalized) ' + header + " (Training over sim_data)")

plt.show()
