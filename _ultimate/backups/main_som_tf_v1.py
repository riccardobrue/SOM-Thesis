import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import som_libs.SOM_TF_2_ext as som_tf

# ---------------------------------------
# PARAMETERS
# ---------------------------------------
epochs = 100
restore_som = True

use_hnd = True  # false-> uses fnd
use_reverse = False  # if true: uses the (trained) som over the network attributes instead of the simulation results
# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------

if use_hnd:
    if use_reverse:
        ckpt_folder = "ok_22x22_hnd_rev_"  # reversed
    else:
        ckpt_folder = "ok_22x22_hnd_"
else:
    if use_reverse:
        ckpt_folder = "ok_22x22_fnd_rev_"  # reversed
    else:
        ckpt_folder = "ok_22x22_fnd_"

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
best_protocols = np.argmax(sim_data_equal[:, [1, 3, 5, 7]], axis=1)  # returns the index of the most efficient protocol

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
lattice_size = heuristic_som_size(all_data.shape[0])  # heuristic lattice size
som_side_dim = int(lattice_size ** .5)  # compute the lattice width - height size
print("SOM dimension: ", som_side_dim, "x", som_side_dim)

# ---------------------------------------
# TRAIN THE SOM
# ---------------------------------------
som = som_tf.SOM(som_side_dim, som_side_dim, all_data.shape[1], epochs=epochs, ckpt_folder_name=ckpt_folder)

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
plt.title('Efficiencies')
plt.bone()  # grayscale colors
# @todo: Which should i take? the transposed or the non-transposed one?
# plt.pcolor(u_matrix)  # plotting the U-MATRIX as background
plt.pcolor(u_matrix.T)  # plotting the transposed U-MATRIX as background (?)
plt.colorbar()
"""
for i in range(0, len(mapped_data)):
    plt.text(mapped_data_X[i], mapped_data_Y[i], best_protocols[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=.5, lw=0))

"""
unique_plot_att = np.unique(best_protocols)

colors = [plt.cm.jet(i / float(len(unique_plot_att) - 1)) for i in range(len(unique_plot_att))]

for i, u in enumerate(unique_plot_att):
    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if best_protocols[j] == u]
    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if best_protocols[j] == u]
    plt.scatter(xi, yi, color=colors[i], label=str(u))
"""
for i, m in enumerate(mapped_data):
    # efficiency values indicates which is the best protocol in the sample (0,1,2,3)
    plt.text(m[1] + .5, m[0] + .5, best_protocols[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=.5, lw=0))

"""
plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(True)
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# FIGURE 2
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(2)
plt.title('Visualizing %AGGR')
plt.bone()  # grayscale colors
plt.pcolor(u_matrix.T)  # plotting the U-MATRIX as background
plt.colorbar()

# mapping based on network topologies
"""
---------------------------
EQUAL - NETWORK ATTRIBUTE INDICES:
---------------------------
'HEIGHT' - 0
'WIDTH' - 1
'NODE' - 2
'R0' - 3
'%AGGR' - 4
'HET' - 5
'HOM ENERGY' - 6
'HOM RATE' - 7
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
att_index = 4

plot_att = net_topology_att_data_equal[:, att_index]
unique_plot_att = np.unique(plot_att)

print("Distinct values: ", plot_att)

markers = ['*', 'o', 'D', 'x', 's', 'd', '.', '+']
colors = ['c', 'g', 'r', 'y', 'm', 'b', 'k', 'w']

# plt.scatter(mapped_data_X, mapped_data_Y, color=colors[index], alpha=.4, marker=markers[index], label=plot_att[index])

#colors = [plt.cm.jet(i / float(len(unique_plot_att) - 1)) for i in range(len(unique_plot_att))]

for i, u in enumerate(unique_plot_att):
    xi = [mapped_data_X[j] for j in range(len(mapped_data_X)) if plot_att[j] == u]
    yi = [mapped_data_Y[j] for j in range(len(mapped_data_Y)) if plot_att[j] == u]
    plt.scatter(xi, yi, color=colors[i], label=str(u))


"""
!!WRONG!!
for i, m in enumerate(mapped_data):
    # plt.text(m[1], m[0], net_topology_att_data_equal[i,4], ha='center', va='center',bbox=dict(facecolor='white', alpha=0.5, lw=0))
    network_att = net_topology_att_data_equal[i, att_index]
    index = np.where(plot_att == network_att)[0][0]
    plt.plot(m[1] + .5, m[0] + .5, color=colors[index], alpha=.4, marker=markers[index], label=plot_att[index])
"""
"""
for i, m in enumerate(mapped_data):
    # network attributes
    # plt.text(m[1], m[0], net_topology_att_data_equal[i,4], ha='center', va='center',bbox=dict(facecolor='white', alpha=0.5, lw=0))
    perc_aggr = net_topology_att_data_equal[i, 4]
    if perc_aggr == 1.:
        plt.plot(m[1], m[0], color="r", alpha=.4, marker="o")
    elif perc_aggr == .7:
        plt.plot(m[1], m[0], color="g", alpha=.4, marker="^")
    elif perc_aggr == .4:
        plt.plot(m[1], m[0], color="c", alpha=.4, marker="s")
    elif perc_aggr == 0.:
        plt.plot(m[1], m[0], color="y", alpha=.4, marker=".")

"""
plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(False)
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
plt.show()
