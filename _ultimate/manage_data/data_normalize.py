import _ultimate.manage_data.merge_data as md
from sklearn import preprocessing


def load_normalized_data(type="equal", shuffle=False):
    # ----------------------------------------------------
    # IMPORTING THE DATA
    # ----------------------------------------------------
    nt, avg_layers, avg_chxrounds, sim, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = md.load_data(
        type=type, shuffle=shuffle)

    avg_layers_shape = avg_layers.shape
    avg_chxrounds_shape = avg_chxrounds.shape
    sim_shape = sim.shape

    # ----------------------------------------------------
    # NORMALIZING THE DATA
    # ----------------------------------------------------
    nt_norm = preprocessing.minmax_scale(nt, feature_range=(0, 1))

    single_avg_layers = avg_layers.ravel()  # flattening of the simulation results to a 1D array for normalization
    single_avg_layers_norm = preprocessing.minmax_scale(single_avg_layers, feature_range=(0, 1))
    avg_layers_norm = single_avg_layers_norm.reshape(avg_layers_shape)

    single_avg_chxrounds = avg_chxrounds.ravel()  # flattening of the simulation results to a 1D array for normalization
    single_avg_chxrounds_norm = preprocessing.minmax_scale(single_avg_chxrounds, feature_range=(0, 1))
    avg_chxrounds_norm = single_avg_chxrounds_norm.reshape(avg_chxrounds_shape)

    single_sim_results = sim.ravel()  # flattening of the simulation results to a 1D array for normalization
    single_sim_results_norm = preprocessing.minmax_scale(single_sim_results, feature_range=(0, 1))
    sim_norm = single_sim_results_norm.reshape(sim_shape)

    return  nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim
