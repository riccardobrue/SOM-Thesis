from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import _ultimate.manage_data.data_normalize as dn


att_index = 4
use_hnd = True  # false-> uses fnd

use_reverse = True  # if true: uses the (trained) som over the network attributes instead of the simulation results

# Axis=1 --> column
# Axis=0 --> row
# ---------------------------------------
# Load the normalized data
# ---------------------------------------
all_data_equal, net_topology_att_data_equal, sim_data_equal, nt_headers_equal, sim_headers_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal, nt_headers_unequal, sim_headers_unequal = dn.load_normalized_unequal_data()

print("=========================================")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)
print("=========================================")

if use_hnd:
    sim_headers = [1, 3, 5, 7]
else:
    sim_headers = [0, 2, 4, 6]

"""
# SOLUTION 0: cluster all together
data = all_data_equal
target = net_topology_att_data_unequal[:, 4]  # 4 --> %aggr 
"""
"""
# SOLUTION 1 : clustering the efficiencies
#data = sim_data_equal[200:600, [0, 1, 2, 3]]
data = sim_data_equal
# target = (sim_data_equal[:, 0] + sim_data_equal[:, 1]) / 2  # efficiency computed as (fnd+hnd)/2
target = net_topology_att_data_unequal[:, 4]  # 4 --> %aggr attribute
# """
# """
# SOLUTION 2: clustering the network topologies
if use_reverse:
    data = net_topology_att_data_equal
else:
    data = sim_data_equal

relevant_targets = sim_data_equal[:, sim_headers]  # select the protocol efficiencies on their hnd value
target = np.argmax(relevant_targets, axis=1)  # gives the index of the maximum value of the efficiency
# """
"""
# SOLUTION 3: clustering the network topologies with target as tipologies
data = net_topology_att_data_equal[:, [4, 5, 6, 7]]  # clusterize the network topology
target = net_topology_att_data_unequal[:, 5]  # 1 --> width attribute
"""
print(data.shape)
print(target.shape)
print(target[500:800])
"""
data=fake_data
target=fake_target

print(data.shape)
print(target.shape)
"""

# -----------------
# data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
# data normalization
# data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, data)

"""
https://stackoverflow.com/questions/19163214/kohonen-self-organizing-maps-determining-the-number-of-neurons-and-grid-size
"""
munits = 5 * data.shape[0] ** 0.54321  # heuristic find the lattice size
som_side_dim = int(munits ** .5)  # compute the lattice size heuristically

big_som_dim = som_side_dim * 4
small_som_dim = som_side_dim * .25

print("SOM dimension: ", som_side_dim)

# Initialization and training
som = MiniSom(som_side_dim, som_side_dim, data.shape[1], sigma=1.0, learning_rate=0.4)
som.random_weights_init(data)
print("Training...")
# som.train_random(data, 10000)  # random training, pick as starting locations from random input vectors
som.train_batch(data, 10000)  # random training
print("\n...ready!")

print(som.get_weights())

u_matrix = som.distance_map()

x = plt.cm.get_cmap('tab10')
colors = x.colors

mapped_data = []
for cnt, xx in enumerate(data):
    mapped_data.append(som.winner(xx))

mapped_data = np.array(mapped_data)
mapped_data_X = mapped_data[:, 0] + .5
mapped_data_Y = mapped_data[:, 1] + .5

best_protocols = np.argmax(sim_data_equal[:, sim_headers], axis=1)  # returns the index of the most efficient protocol

best_protocols_names = sim_headers_equal[sim_headers]

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
