import tensorflow as tf
import numpy as np

class SOM(object):
    """
    A 2-D Kohonen Map (more commonly known as a self-organizing map, or SOM)
    Neighbourhood function: Gaussian
    Learning Rate: decreases linearly.
    """

    def __init__(self, m, n, dim, n_iterations=100, alpha=0.05, sigma=None):
        """
        m X n = dimensions of the SOM.
        n_iterations = number of training iterations
        dim = dimensionality of the training inputs.
        alpha = float value for initial learning rate.
        sigma = radius of influence of the best matching unit (BMU).
                default value is half of max(m, n)
        """

        # Assign required variables
        self._m = m
        self._n = n
        self._centroid_grid = []  # initialize centroid grid to observe training
        self._map_vectors = []  # initialize map vectors, which store topography
        self._n_iterations = int(n_iterations)

        alpha = float(alpha)
        if sigma is None:  # setting default neighborhood radius
            sigma = max(m, n) / 2.2
        else:
            sigma = float(sigma)

        # Init our TensorFlow graph
        self._graph = tf.Graph()

        # as_default allows us to assign operations and tensors in "self"
        with self._graph.as_default():

            # Assign random weights for nodes in grid and
            # store as a matrix of size [m*n, dim]
            self._weight_vects = tf.Variable(tf.random_normal(
                [m * n, dim]))

            # Create matrix of size [m*n, 2] for SOM grid
            self._location_vects = tf.constant(np.array(
                list(self._node_locations(m, n))))

            # Training vector TF placeholder
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration TF placeholder
            self._iter_input = tf.placeholder("float")

            # Compute the Best Matching Unit given a vector using Euclidean distance
            # between every node's weight vector and the input
            # Returns the index of the node which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.sub(self._weight_vects, tf.pack(
                    [self._vect_input for i in range(m * n)])), 2), 1)), 0)

            # Gets location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))), [2])

            # Compute alpha and sigma based on iteration number
            learning_rate_op = tf.sub(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.mul(alpha, learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning
            # rates for all nodes, based on iteration number and location.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.sub(
                self._location_vects, tf.pack(
                    [bmu_loc for i in range(m * n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weight vectors of all nodes based on a particular input
            learning_rate_multiplier = tf.pack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                for i in range(m * n)])
            weight_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(tf.pack([self._vect_input for i in range(m * n)]),
                       self._weight_vects))
            new_weights_op = tf.add(self._weight_vects,
                                    weight_delta)
            self._training_op = tf.assign(self._weight_vects,
                                          new_weights_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)

    def _node_locations(self, m, n):
        """
        Yields the 2-D locations of the individual nodes in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' is an iterable of 1-D NumPy arrays.
        Uses random weight vectors as starting conditions for training.
        """

        # Count training iterations. Process an iteration
        for iter_no in range(self._n_iterations):

            # Take all input vectors and train using one at a time.
            for input_vect in input_vects:
                self._sess.run(self._training_op, feed_dict={self._vect_input: input_vect, self._iter_input: iter_no})

            # Save each calculated centroid location to a grid.
            centroid_grid = [[] for i in range(self._m)]
            self._weights = list(self._sess.run(self._weight_vects))
            self._locations = list(self._sess.run(self._location_vects))
            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weights[i])
            self._centroid_grid.append(centroid_grid)  # store the centroid grid to list of
            # previous grids for reference

            self._to_map = []
            for vector in input_vects:
                min_index = min([i for i in range(len(self._weights))],
                                key=lambda x: np.linalg.norm(vector - self._weights[x]))

                self._to_map.append(self._locations[min_index])
            self._map_vectors.append(self._to_map)

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant node in the SOM grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped node.
        """
        # return self._map_vectors

        # [self._vect_input for i in range(m*n)])), 2), 1)),0)

        # This will extract the location of the BMU based on the BMU's index
        slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]))), [2])

        # To compute the alpha and sigma values based on iteration number
        learning_rate_op = tf.sub(1.0, tf.div(self._iter_input, self._n_iterations))
        _alpha_op = tf.mul(alpha, learning_rate_op)
        _sigma_op = tf.mul(sigma, learning_rate_op)

        # Construct the op that will generate a vector with learning
        # rates for all neurons, based on iteration number and location
        # wrt BMU.
        bmu_distance_squares = tf.reduce_sum(
            tf.pow(tf.sub(self._location_vects, tf.pack([bmu_loc for i in range(m * n)])), 2), 1)
        neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
        learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)

        # Finally, the op that will use learning_rate_op to update
        # the weightage vectors of all neurons based on a particular
        # input
        learning_rate_multiplier = tf.pack(
            [tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m * n)])
        weightage_delta = tf.mul(learning_rate_multiplier,
                                 tf.sub(tf.pack([self._vect_input for i in range(m * n)]), self._weightage_vects))
        new_weightages_op = tf.add(self._weightage_vects, weightage_delta)
        self._training_op = tf.assign(self._weightage_vects, new_weightages_op)

        ##INITIALIZE SESSION
        self._sess = tf.Session()

        ##INITIALIZE VARIABLES
        init_op = tf.initialize_all_variables()
        self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op, feed_dict={self._vect_input: input_vect, self._iter_input: iter_no})
            print("Iteration %d complete" % iter_no)

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order)
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return
