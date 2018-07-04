import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import som_libs.SOM_TF_2_ext as som_tf
from matplotlib.pyplot import cm

# ---------------------------------------
# PARAMETERS
# ---------------------------------------
# epochs = 80
epochs = 100
restore_som = True

use_hnd = True  # false-> uses fnd

ckpt_folder = "ok_22x22_fnd_"
# ckpt_folder = "ok_22x22_hnd_"
# ckpt_folder = "ok_30x30_hnd_"  # this is made with 80 epochs

heuristic_size = True  # 22x22
# som_side_dim = 30

# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------
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
print("Best protocols: ", best_protocols)

# ---------------------------------------
# SELECT THE DATA FOR CLUSTERING
# ---------------------------------------
all_data = sim_data_equal
# all_data = net_topology_att_data_equal
# all_data = all_data_equal
print("Clustering data size: ", all_data.shape)

# ---------------------------------------
# COMPUTE THE SOM SIZE HEURISTICALLY
# ---------------------------------------
if heuristic_size:
    lattice_size = heuristic_som_size(all_data.shape[0])  # heuristic lattice size
    som_side_dim = int(lattice_size ** .5)  # compute the lattice width - height size
print("SOM dimension: ", som_side_dim, "x", som_side_dim)

# ---------------------------------------
# TRAIN THE SOM
# ---------------------------------------
som = som_tf.SOM(som_side_dim, som_side_dim, all_data.shape[1], n_iterations=epochs, checkpoint_folder_name=ckpt_folder)

if restore_som:
    som.restore()
if train_som:
    print("Training the SOM:")
    # fix or find another system to continue the learning from a certain point (with reduced learning_rate)
    som.train(all_data, restart_from=0)
    # restart_from indicates the iteration number
    # (to decrease the learning rate after a restored session)

# ---------------------------------------
# GET THE OUTPUT GRID AND COMPUTE U-MATRIX
# ---------------------------------------
image_grid = som.get_centroids()
mat = np.zeros(shape=(som_side_dim, som_side_dim, all_data.shape[1]))
for r in range(0, len(image_grid)):
    for c in range(0, len(image_grid[r])):
        mat[r][c] = image_grid[r][c]
u_matrix = distance_map(mat)

# ---------------------------------------
# CREATE A MAPPED DATA TO KNOW THE BMUs OVER THE DATA
# ---------------------------------------
mapped_data = np.array(som.map_vects(all_data))
mapped_data_X = mapped_data[:, 0] + .5
mapped_data_Y = mapped_data[:, 1] + .5
print("Mapped data: ", mapped_data_X)

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
plt.title('Best protocols')
plt.bone()  # grayscale colors
# @todo: Which should i take? the transposed or the non-transposed one?
# plt.pcolor(u_matrix)  # plotting the U-MATRIX as background
plt.pcolor(u_matrix.T)  # plotting the transposed U-MATRIX as background (?)
plt.colorbar()
print(best_protocols_names)
classes = best_protocols
unique_classes = np.unique(classes)

# markers = ['*', 'o', 'D', 'x', 's', 'd', '.', '+']
# create one color and one mark for each class
x = plt.cm.get_cmap('tab10')
colors = x.colors

for i, u in enumerate(unique_classes):
    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if classes[j] == u]
    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if classes[j] == u]
    plt.scatter(xi, yi, color=colors[i], label=str(u) + " Protocol won", alpha=.5)

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(True)
plt.legend(loc='center left', bbox_to_anchor=(1.18, 0.5))
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# FIGURE 2
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(2)
plt.bone()  # grayscale colors
plt.pcolor(u_matrix.T)  # plotting the U-MATRIX as background
plt.colorbar()

"""
---------------------------
EQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
'HEIGHT' - 0
'WIDTH' - 1
'NODE' - 2
'R0' - 3
'%AGGR' - 4 - important
'HET' - 5
'HOM ENERGY' - 6
'HOM RATE' - 7 - important
---------------------------
EQUAL - PROTOCOLS INDICES:
---------------------------
'REECHD FND' - 0 
'REECHD HND' - 1
'HEED FND' - 2
'HEED HND' - 3
'ERHEED FND' - 4
'ERHEED HND' - 5
'FMUC FND' - 6
'FMUC HND' - 7
---------------------------
UNEQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
---------------------------
UNEQUAL - PROTOCOLS INDICES:
---------------------------
"""
# mapping based on network topologies
att_index = 7
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
    plt.scatter(xi, yi, color=colors[i], label=str(u) + " " + header, alpha=.5)

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(False)
plt.legend(loc='center left', bbox_to_anchor=(1.18, 0.5))
plt.title('Visualizing ' + header)
plt.show()
