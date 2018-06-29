import merge_data as md
import numpy as np
from sklearn import preprocessing

# ----------------------------------------------------
# IMPORTING THE DATA
# ----------------------------------------------------

network_topology_att, simulation_results, nt_headers, sim_headers = md.load_data()

print("===================")
print(nt_headers)
print("-------------------")
print(network_topology_att)
print("===================")
print(sim_headers)
print("-------------------")
print(simulation_results)
print("===================")
sim_results_shape = simulation_results.shape
print("Network topology attributes (size): ", network_topology_att.shape)
print("Simulation results (size): ", sim_results_shape)
print("===================")

# ----------------------------------------------------
# NORMALIZING THE DATA
# ----------------------------------------------------
print("Normalization...")
print("===================")

max_sim_result = np.amax(simulation_results)
min_sim_result = np.amin(simulation_results)
print("Sim max: ", max_sim_result)
print("Sim min: ", min_sim_result)

network_topology_att_norm = preprocessing.minmax_scale(network_topology_att, feature_range=(0, 1))
# METHOD 1: each column is normalized in an independent way
# simulation_results_norm = preprocessing.minmax_scale(simulation_results, feature_range=(0, 1))
# METHOD 2: all the simulation outputs are linked together while normalizing because they express the same concept
single_sim_results = simulation_results.ravel()  # flattening of the simulation results to a 1D array for normalization
single_sim_results_norm = preprocessing.minmax_scale(single_sim_results, feature_range=(0, 1))
simulation_results_norm = single_sim_results_norm.reshape(sim_results_shape)

print("===================")
print(nt_headers)
print("-------------------")
print(network_topology_att_norm)
print("===================")
print(sim_headers)
print("-------------------")
print(simulation_results_norm)
print("===================")
print("Network topology attributes (size): ", network_topology_att_norm.shape)
print("Simulation results (size): ", simulation_results_norm.shape)
print("===================")
