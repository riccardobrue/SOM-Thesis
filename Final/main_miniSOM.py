from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import manage_data.data_normalize as dn
import utilities

# ---------------------------------------
# LOAD THE NORMALIZED DATA
# ---------------------------------------
all_data_equal, net_topology_att_data_equal, sim_data_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal = dn.load_normalized_unequal_data()

print("=========================================")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)
print("=========================================")

# ----------------------------------------------
# SELECTING THE RIGHT DATA
# ----------------------------------------------

"""
# SOLUTION 1: clustering all the data together
data = all_data_equal
target = net_topology_att_data_equal[:, 4]  # 4 --> %aggr 
"""
"""
# SOLUTION 2: clustering the efficiencies
#data = sim_data_equal[200:600, [0, 1, 2, 3]] #just a subset from all the simulation data
data = sim_data_equal
target = net_topology_att_data_equal[:, 4]  # 4 --> %aggr attribute
# """
# """
# SOLUTION 3: clustering the network topologies
data = net_topology_att_data_equal  # clustering the network topologies
relevant_targets = sim_data_equal[:, [1, 3, 5, 7]]  # select the protocol efficiencies on their hnd value
target = np.argmax(relevant_targets, axis=1)  # gives the index of the maximum value of the efficiency
# """
"""
# SOLUTION 4: clustering the network topologies with target as topologies
data = net_topology_att_data_equal[:, [4, 5, 6, 7]]  # clustering the network topology
target = net_topology_att_data_unequal[:, 1]  # 1 --> width attribute
"""

print(data.shape)
print(target.shape)
print(target[500:800])

"""
https://stackoverflow.com/questions/19163214/kohonen-self-organizing-maps-determining-the-number-of-neurons-and-grid-size
"""
munits = utilities.mapunits(data.shape[0])  # heuristic lattice size
som_dim = int(munits ** .5)  # compute the lattice width - height size heuristically

print("SOM's side dimension: ", som_dim)

# Initialization and training
som = MiniSom(som_dim, som_dim, data.shape[1], sigma=1.0, learning_rate=0.4)
som.random_weights_init(data)
print("Training...")

# som.train_random(data, 5000)  # random training
som.train_batch(data, 5000)  # random training

print("...ready!")

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
        print(w)
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markerfacecolor='None',  # instead of target use t
                 markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)  # instead of target use t
    except():
        pass

plt.axis([0, som_dim, 0, som_dim])
plt.show()
print("SOM's weights")
print(som.get_weights())
