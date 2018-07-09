import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import _ultimate.manage_data.data_normalize as dn
import _ultimate.manage_data.merge_data as md
from matplotlib import pyplot as plt
import numpy as np
import _ultimate.som_libs.SOM_TF_2_ext as som_tf
import os
import scipy


# ---------------------------------------
# COMPUTING THE U-MATRIX
# ---------------------------------------
def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(np.dot(x, x.T))


def distance_map(weights):
    # weights is a 22x22x8 --> som_dim x som_dim x input_features
    """Returns the distance map of the weights.
    Each cell is the normalised sum of the distances between
    a neuron and its neighbours."""
    um = np.zeros((weights.shape[0], weights.shape[1]))  # 22x22 filled with 0
    it = np.nditer(um, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):  # add 1 column before and 1 after
            for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):  # add 1 row up and 1 down
                if 0 <= ii < weights.shape[0] and 0 <= jj < weights.shape[1]:
                    w_1 = weights[ii, jj, :]
                    w_2 = weights[it.multi_index]
                    um[it.multi_index] += fast_norm(w_1 - w_2)
        it.iternext()
    um = um / um.max()
    return um


class SOM:
    def __init__(self, shape, input_data):
        assert isinstance(shape, (int, list, tuple))
        assert isinstance(input_data, (list, np.ndarray))
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        input_shape = tuple(input_data.shape)
        assert 2 == len(input_shape)
        self.shape = tuple(shape)
        self.input_layer = input_data
        self.input_num = input_shape[0]
        self.input_dim = input_shape[1]
        self.output_layer = rand.standard_normal((self.shape[0] * self.shape[1], self.input_dim))
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        self.index_map = np.hstack((y.flatten()[:, np.newaxis],
                                    x.flatten()[:, np.newaxis]))
        self._param_input_length_ratio = 0.25
        self._life = self.input_num * self._param_input_length_ratio
        self._param_neighbor = 0.25
        self._param_learning_rate = 0.1

    def set_parameter(self, neighbor=None, learning_rate=None, input_length_ratio=None):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def set_default_parameter(self, neighbor=0.25, learning_rate=0.1, input_length_ratio=0.25):
        if neighbor:
            self._param_neighbor = neighbor
        if learning_rate:
            self._param_learning_rate = learning_rate
        if input_length_ratio:
            self._param_input_length_ratio = input_length_ratio
            self._life = self.input_num * self._param_input_length_ratio

    def _get_winner_node(self, data):
        sub = self.output_layer - data
        dis = np.linalg.norm(sub, axis=1)
        bmu = np.argmin(dis)
        return np.unravel_index(bmu, self.shape)

    def _update(self, bmu, data, i):
        dis = np.linalg.norm(self.index_map - bmu, axis=1)
        L = self._learning_rate(i)
        S = self._learning_radius(i, dis)
        self.output_layer += L * S[:, np.newaxis] * (data - self.output_layer)

    def _learning_rate(self, t):
        return self._param_learning_rate * np.exp(-t / self._life)

    def _learning_radius(self, t, d):
        s = self._neighbourhood(t)
        return np.exp(-d ** 2 / (2 * s ** 2))

    def _neighbourhood(self, t):
        initial = max(self.shape) * self._param_neighbor
        return initial * np.exp(-t / self._life)

    def train(self, n):
        for i in range(n):
            r = rand.randint(0, self.input_num)
            data = self.input_layer[r]
            win_idx = self._get_winner_node(data)
            self._update(win_idx, data, i)
        # return self.output_layer.reshape(self.shape + (self.input_dim,))
        return self.output_layer.reshape((self.shape[1], self.shape[0], self.input_dim))





# ---------------------------------------
# PARAMETERS
# ---------------------------------------
# ----------------
# training-restoring parameters
# ----------------

folder_prefix = "ok_ext_"
epochs = 5000

restore_som = True  # true: doesn't train the som and doesn't store any new checkpoint files

heuristic_size = True  # 22x22 (if false it is needed to specify the "som_side_dim" variable and the "ckpt_folder" name)
manually_picked_som_dim = 30  # if heuristic_size is False, this will be the chosen som's side size

use_reverse = True  # if true: uses the (trained) som over the network attributes instead of the simulation results

use_hnd = False  # false-> uses fnd

# ---------------------------------------
# DERIVED PARAMETERS
# ---------------------------------------
if heuristic_size:
    ckpt_folder_size_name = "22x22"
else:
    ckpt_folder_size_name = str(manually_picked_som_dim) + "x" + str(manually_picked_som_dim)

if use_reverse:
    ckpt_folder = folder_prefix + ckpt_folder_size_name + "_rev_"  # reversed
else:
    if use_hnd:
        ckpt_folder = folder_prefix + ckpt_folder_size_name + "_hnd_"
    else:
        ckpt_folder = folder_prefix + ckpt_folder_size_name + "_fnd_"

ckpt_folder = ckpt_folder + str(epochs)  # the folder name is composed by "cpkt_folder" string + epochs number

train_som = not restore_som
store_som = not restore_som



if __name__ == '__main__':
    #    data = np.array([[0, 0, 0], [1, 1, 1]])
    #    som = SOM((10, 10), data)
    #    map = som.train(20)
    #    #print map
    #    plt.imshow(map)
    #    plt.show()
    N = 30
    # input = np.random.rand(10000, 3)
    # input = [[0,1,0],[1,0,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]
    inpt = [[0, 0, 0], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05], [0, 1, 0], [0.95, 0.95, 0.95], [1, 1, 1], [0.9, 0.9, 0.9]]

    # ---------------------------------------
    # LOAD THE NORMALIZED DATA
    # ---------------------------------------
    nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
        type="equal")

    headers_nt = headers_nt[3:]
    nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

    som = SOM((N, N), nt_norm)
    #som = SOM((N, N), inpt)


    som.set_parameter()
    # p = plt.imshow(map0, interpolation='none')
    ims = []
    #    w = cv2.VideoWriter('/home/yota/Pictures/output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, (N, N))
    for i in range(500):
        m = som.train(10)
        m=m.tolist()
        print(i)

        #_get_winner_node()


        mat = np.zeros(shape=(N, N, nt_norm.shape[1]))
        for r in range(0, len(m)):
            for c in range(0, len(m[r])):
                mat[r][c] = m[r][c]
        u_matrix = distance_map(mat)


        #img = np.array(m, dtype=np.uint8)
        #        w.write(img)
        im = plt.imshow(u_matrix, interpolation='none')  # , animated=True)
        ims.append([im])
    #    w.release()
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    #    ani.save('/home/yota/Pictures/dynamic_images.mp4')
    #    ani.save('dynamic_images.mp4')
    #    plt.subplot(2, 2, 1)
    #    plt.imshow(map0, interpolation='none')
    #    plt.subplot(2, 2, 2)
    #    plt.imshow(map1, interpolation='none')
    #    plt.subplot(2, 2, 3)
    #    plt.imshow(map2, interpolation='none')
    #    plt.subplot(2, 2, 4)
    #    plt.imshow(map3, interpolation='none')
    plt.show()
