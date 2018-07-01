from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import data_normalize as dn

# ---------------------------------------
# Load the normalized data
# ---------------------------------------
all_data_equal, net_topology_att_data_equal, sim_data_equal = dn.load_normalized_equal_data()
all_data_unequal, net_topology_att_data_unequal, sim_data_unequal = dn.load_normalized_unequal_data()

print("=========================================")
print("Equal size: ", all_data_equal.shape)
print("Unequal size: ", all_data_unequal.shape)
print("=========================================")
# print(net_topology_att_data_equal)


data = sim_data_equal[:, [0, 1]]
target = (sim_data_equal[:, 0] + sim_data_equal[:, 1]) / 2  # efficiency computed as (fnd+hnd)/2

print(target)

# data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
# data normalization
data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, data)

# Initialization and training
som = MiniSom(7, 4, data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
print("Training...")
som.train_random(data, 100)  # random training
print("\n...ready!")

# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()

t = np.zeros(len(target), dtype=int)
t[target < .25] = 0
t[target >= .25] = 1
t[target >= .28] = 2

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0] + .5, w[1] + .5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.show()
