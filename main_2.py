import data_normalize as dn
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from SOM import SOM
import numpy as np
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
# IMPLEMENT THE SOM
# ---------------------------------------

all_data=sim_data_equal
labels=net_topology_att_data_equal[:,1]


# Train a 20x30 SOM (2D) with 400 iterations
# 20: SOM's grid height
# 30: SOM's grid width
# 3: input dimensionality (RGB values)
# 400: iterations
som = SOM(10, 10, all_data.shape[1], 10)
som.train(all_data, training_graph=False, labels=labels)  # training the SOM with the input data

# Get output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(all_data)

print("Plotting final chart")
# Plot
plt.imshow(image_grid)



print(image_grid)
print("------------")
print(mapped)

# Plotting the response for each pattern in the iris dataset
#plt.bone()
#plt.pcolor(image_grid)  # plotting the distance map as background (u-matrix)
#plt.colorbar()

plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], labels[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.show()

