import manage_data.data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_som.SOM_TF_1 as som_tf
import utilities


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
                if (ii >= 0 and ii < weights.shape[0] and
                        jj >= 0 and jj < weights.shape[1]):
                    w_1 = weights[ii, jj, :]
                    w_2 = weights[it.multi_index]
                    um[it.multi_index] += fast_norm(w_1 - w_2)
        it.iternext()
    um = um / um.max()
    return um


# ---------------------------------------
# Load the normalized data
# ---------------------------------------
all_data_equal, net_topology_att_data_equal, sim_data_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal = dn.load_normalized_unequal_data()

print("=========================================")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)
print("=========================================")

# ---------------------------------------
# IMPLEMENT THE SOM WITH TENSORFLOW
# ---------------------------------------
#all_data = sim_data_equal
#all_data = net_topology_att_data_equal
all_data = all_data_equal
print(all_data.shape)

# ----------------------------------------------------------------------------------------------------------------------

# Create a permutation of the original columns
# shuffle_index = np.random.choice(range(len(colors)), size=15, replace=False)
# shuffle_colors = np.array(colors)[shuffle_index]
# shuffle_color_names = np.array(color_names)[shuffle_index]

# combined_colors = [list(x) + colors[i] for i, x in enumerate(shuffle_colors)]

# print(combined_colors)

# Setup the SOM object
# som = SOM(m=20, n=30, dim=6, n_iterations=400)

munits = utilities.mapunits(all_data.shape[0])  # heuristic lattice size
som_dim = int(munits ** .5)  # compute the lattice width - height size heuristically

print("SOM's side dimension: ", som_dim)

som = som_tf.SOM(som_dim, som_dim, all_data.shape[1], n_iterations=5)
som.train(all_data)

# Train on the new colors
# som.train(combined_colors)

# Get output grid
image_grid = som.get_centroids()
# Map colours to their closest neurons
# mapped = som.map_vects(combined_colors)
mapped = som.map_vects(net_topology_att_data_equal[:, 1])

# print(image_grid)
# print("-")
# print(image_grid[0])
# print("-")
# print(image_grid[0][0])

mat = np.zeros(shape=(som_dim, som_dim, all_data.shape[1]))

# print("------------")
for r in range(0, len(image_grid)):
    for c in range(0, len(image_grid[r])):
        # print(image_grid[r][c])
        mat[r][c] = image_grid[r][c]
        # print("--")
    # print("------")
print("------------")

print(mat)
print(mat.shape)

distances=distance_map(mat)
print(distances)
print(distances.shape)
# compute:
# u-matrix,
# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(distances.T)  # plotting the distance map as background
plt.colorbar()
plt.show()


relevant_targets = sim_data_equal[:, [1, 3, 5, 7]]  # select the protocol efficiencies on their hnd value
target = np.argmax(relevant_targets, axis=1)  # gives the index of the maximum value of the efficiency

t = np.zeros(len(target), dtype=int)
t[target == 0.] = 0
t[target == 1.] = 1
t[target == 2.] = 2
t[target == 3.] = 3

markers = ['o', 's', '.', '^']
colors = ['g', 'r', 'b', 'y']

"""
for cnt, xx in enumerate(all_data):
    try:
        w = som.winner(xx)  # getting the winner
        print(w)
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markerfacecolor='None',  # instead of target use t
                 markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)  # instead of target use t
    except():
        pass
"""
plt.axis([0, som_dim, 0, som_dim])
plt.show()
"""
https://stackoverflow.com/questions/25258191/how-plot-u-matrix-sample-hit-and-input-planes-from-a-trained-data-by-som
https://stackoverflow.com/questions/21203823/formulation-of-the-u-matrix-unified-distance-matrix-as-a-matrix-operation
"""

"""
print("---------------------")
print("---------------------")
print("----------------")
print(image_grid)
print("----------------")
print(mapped)
print("---------------------")
print("---------------------")
# Now that we have the trained SOM, we are going to extract the
s_min = 0
s_max = 5
s_size = 3
init_slice = np.array(np.arange(s_min, s_min + s_size))
max_slice = s_max - s_min - s_size + 2
slicing_array = [init_slice + x for x in range(max_slice)]

# Because the data is now 6-dimensional, we cannot plot it immediately, hence we need to slice it.
list_image_grid_sel = []
for i_slice in slicing_array:
    dum = [x[i_slice] for b in image_grid for x in b]
    list_image_grid_sel.append(np.reshape(dum, (20, 30, 3)))
print("-----")
for i_plot, i_data in enumerate(list_image_grid_sel):
    plt.figure(i_plot)
    plt.imshow(i_data)
    plt.show()
"""
