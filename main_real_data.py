import merge_data as md
import numpy as np

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
print("Input size: ", network_topology_att.shape)
print("Output size: ", simulation_results.shape)
print("===================")

# ----------------------------------------------------
# MANAGING THE DATA
# ----------------------------------------------------
