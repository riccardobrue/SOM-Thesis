import data_normalize as dn
from minisom import MiniSom
import matplotlib.pylab as plt
from sklearn import datasets
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
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
print(X)
all_data = sim_data_equal[:,1:6]  # [500:1500]
#all_data=X
print(all_data)
# @todo
# add a further "output" column stating which is the winner (for labelling purposes)


"""
https://github.com/JustGlowing/minisom
https://github.com/JustGlowing/minisom/blob/master/examples/examples.ipynb
"""
som = MiniSom(12, 12, all_data.shape[1], sigma=0.2, learning_rate=0.4)  # initialization of 30x30 SOM

# Initialization and training
som.random_weights_init(all_data)
print("Training...")
som.train_random(all_data, 500)  # trains the SOM with 100 iterations
print("...ready!")

to_print=som.distance_map().T
print(to_print.shape)
print(to_print)


# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background (u-matrix)
plt.colorbar()

"""
target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
t = np.zeros(len(target), dtype=int)
t[target == 'setosa'] = 0
t[target == 'versicolor'] = 1
t[target == 'virginica'] = 2

# use different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
"""
plt.show()
