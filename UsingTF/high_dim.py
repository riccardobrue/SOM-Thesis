import data_normalize as dn
from matplotlib import pyplot as plt
import numpy as np
from SOM_TF import SOM

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
all_data = sim_data_equal
print(all_data.shape)

# ----------------------------------------------------------------------------------------------------------------------
# Training inputs for RGBcolors
colors = np.array(
    [[0., 0., 0., 0., 1., 0.],
     [0., 0., 1., 0., 0., 0.5],
     [0., 0., 0.5, 0., 0., 0.],
     [0.125, 0.529, 1.0, 0.5, 0., 0.],
     [0.33, 0.4, 0.67, 0., 0.5, 0.],
     [0.6, 0.5, 1.0, 0., 0., 1],
     [0., 1., 0., 1, 0., 0.],
     [1., 0., 0., 1., 0.6, 0.],
     [0., 1., 1., 0., 0.6, 0.],
     [1., 0., 1., 0.8, 0., 0.6],
     [1., 1., 0., 0., 0.8, 0.],
     [1., 1., 1., 0.8, 0., 0.],
     [.33, .33, .33, 0., 0., 0.],
     [.5, .5, .5, 0., 0.8, 0.2],
     [.66, .66, .66, 0., 1., 0.2]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# Create a permutation of the original columns
# shuffle_index = np.random.choice(range(len(colors)), size=15, replace=False)
# shuffle_colors = np.array(colors)[shuffle_index]
# shuffle_color_names = np.array(color_names)[shuffle_index]

# combined_colors = [list(x) + colors[i] for i, x in enumerate(shuffle_colors)]

# print(combined_colors)

# Setup the SOM object
# som = SOM(m=20, n=30, dim=6, n_iterations=400)

som = SOM(20, 30, all_data.shape[1], 400)
som.train(all_data)

# Train on the new colors
# som.train(combined_colors)

# Get output grid
image_grid = som.get_centroids()
# Map colours to their closest neurons
# mapped = som.map_vects(combined_colors)
mapped = som.map_vects(colors)

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
