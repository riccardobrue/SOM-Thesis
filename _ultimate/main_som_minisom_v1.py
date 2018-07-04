from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import data_normalize as dn

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
#"""
#SOLUTION 2: clustering the network topologies
data = net_topology_att_data_equal #clusterize the network topology
relevant_targets = sim_data_equal[:, [1, 3, 5, 7]]  # select the protocol efficiencies on their hnd value (three protocols)
target = np.argmax(relevant_targets, axis=1)  # gives the index of the maximum value of the efficiency
#"""
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
som_dim = int(munits ** .5)  # compute the lattice size heuristically

big_som_dim = som_dim * 4
small_som_dim = som_dim * .25

print("SOM dimension: ", som_dim)

# Initialization and training
som = MiniSom(som_dim, som_dim, data.shape[1], sigma=1.0, learning_rate=0.4)
som.random_weights_init(data)
print("Training...")
#som.train_random(data, 10000)  # random training, pick as starting locations from random input vectors
som.train_batch(data, 5000)  # random training
print("\n...ready!")

# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()

t = np.zeros(len(target), dtype=int)
t[target == 0.] = 0
t[target == 1.] = 1
t[target == 2.] = 2
t[target == 3.] = 3

# use different colors and markers for each label
markers = ['o', 's', '.', '^']
colors = ['g', 'r', 'b', 'y']

for cnt, xx in enumerate(data):
    try:
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markerfacecolor='None',  # instead of target use t
                 markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)  # instead of target use t
    except:
        pass

plt.axis([0, som_dim, 0, som_dim])
plt.show()

print(som.get_weights())
