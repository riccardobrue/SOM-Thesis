# For plotting the images
from matplotlib import pyplot as plt
from SOM import SOM
import numpy as np

# Training inputs for RGB colors (percentage of Red, Green and Blue)
colors = np.array(
    [[0., 0., 0.],  # black
     [0., 0., 1.],  # blue
     [0., 0., 0.5],  # dark blue
     [0.125, 0.529, 1.0],  # sky blue
     [0.33, 0.4, 0.67],  # grey blue
     [0.6, 0.5, 1.0],  # lilac
     [0., 1., 0.],  # green
     [1., 0., 0.],  # red
     [0., 1., 1.],  # cyan
     [1., 0., 1.],  # violet
     [1., 1., 0.],  # yellow
     [1., 1., 1.],  # white
     [.33, .33, .33],  # dark grey
     [.5, .5, .5],  # medium grey
     [.66, .66, .66]  # light grey
     ])

color_names = ['black', 'blue', 'darkblue', 'skyblue',
               'greyblue', 'lilac', 'green', 'red',
               'cyan', 'violet', 'yellow', 'white',
               'darkgrey', 'mediumgrey', 'lightgrey']

# Train a 20x30 SOM (2D) with 400 iterations
# 20: SOM's grid height
# 30: SOM's grid width
# 3: input dimensionality (RGB values)
# 400: iterations
som = SOM(20, 30, 3, 2000)
som.train(colors)  # training the SOM with the input data

# Get output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(colors)

# Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.show()
