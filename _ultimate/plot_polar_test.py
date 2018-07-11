from matplotlib import pyplot as plt
import numpy as np

# load the matrix which indicates the neuron vectors
mat = np.load("mat.npy")

# create an array of colors
x = plt.cm.get_cmap('tab10')
colors = x.colors

print('Plt Backend:', plt.get_backend())

f, axarr = plt.subplots(mat.shape[0], mat.shape[1], subplot_kw=dict(projection='polar'))

for rr in range(0, mat.shape[0]):
    for cc in range(0, mat.shape[1]):
        print(rr, " ", cc)

        N = mat.shape[2]  # number of features -> mat.shape[2]
        theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)  # direction of the bar around the circle
        radii = mat[rr, cc] * 10  # length of the bar (radius) based on the value inside the matrix of the data [0->1]
        width = np.pi / N  # width of the bar

        bars = axarr[rr, cc].bar(theta, radii, width=width, bottom=0.0)  # create the bar chart, each bar is a feature
        ii = 0
        for r, bar in zip(radii, bars):  # iterate the bars, each bar is a feature
            bar.set_facecolor(colors[ii])  # pick the color for the bar
            bar.set_alpha(0.8)  # set the transparency
            ii = ii + 1

        axarr[rr, cc].set_yticklabels([])  # do not show the y labels of the subplots
        axarr[rr, cc].set_xticklabels([])  # do not show the x labels of the subplots
        axarr[rr, cc].set_title('')  # do not show the title of the subplots

f.subplots_adjust(hspace=0, wspace=0)  # set the distances between the sub charts

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
#plt.show()
plt.savefig("test.png",bbox_inches='tight')
