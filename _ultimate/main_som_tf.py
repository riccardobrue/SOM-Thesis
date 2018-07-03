import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import som_libs.SOM_TF_2_ext as som_tf

# ---------------------------------------
# PARAMETERS
# ---------------------------------------
epochs = 5
restore_som = True
ckpt_folder = "ok_"

# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------
train_som = not restore_som
store_som = not restore_som
ckpt_folder = ckpt_folder + str(epochs)


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
all_data_equal, net_topology_att_data_equal, sim_data_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal = dn.load_normalized_unequal_data()
# Get the best protocol for each row
best_protocols = np.argmax(sim_data_equal[:, [1, 3, 5, 7]], axis=1)  # returns the index of the most efficient protocol

print("Data loaded: ")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)

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
mapped_data = som.map_vects(sim_data_equal)

# ---------------------------------------
# STORE THE LEARNED SOM AND CLOSE THE TENSORFLOW SESSION
# ---------------------------------------
if store_som:
    som.store()
som.close_sess()

# ---------------------------------------
# VISUALIZING THE CHARTS
# ---------------------------------------
plt.figure(1)
plt.title('Efficiencies')
# plt.bone() #grayscale colors
plt.pcolor(u_matrix.T)  # plotting the U-MATRIX as background
plt.colorbar()

for i, m in enumerate(mapped_data):
    # efficiency values indicates which is the best protocol in the sample (0,1,2,3)
    plt.text(m[1] + .5, m[0] + .5, best_protocols[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=.5, lw=0))

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(True)
plt.show()

plt.figure(2)
plt.title('Visualizing %AGGR')
# plt.bone() #grayscale colors
plt.pcolor(u_matrix.T)  # plotting the U-MATRIX as background
plt.colorbar()

# mapping based on network topologies
for i, m in enumerate(mapped_data):
    # plt.text(m[1], m[0], net_topology_att_data_equal[i,4], ha='center', va='center',bbox=dict(facecolor='white', alpha=0.5, lw=0))
    perc_aggr = net_topology_att_data_equal[i, 4]
    if perc_aggr == 1.:
        plt.plot(m[1] + .5, m[0] + .5, color="r", alpha=.4, marker="o")
    elif perc_aggr == .7:
        plt.plot(m[1] + .5, m[0] + .5, color="g", alpha=.4, marker="^")
    elif perc_aggr == .4:
        plt.plot(m[1] + .5, m[0] + .5, color="c", alpha=.4, marker="s")
    elif perc_aggr == 0.:
        plt.plot(m[1] + .5, m[0] + .5, color="y", alpha=.4, marker=".")

plt.axis([0, som_side_dim, 0, som_side_dim])
plt.interactive(False)
plt.show()
