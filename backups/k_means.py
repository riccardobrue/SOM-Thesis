import numpy as np
import matplotlib.pyplot as plt
import _ultimate.manage_data.data_normalize as dn
from sklearn.cluster import KMeans

nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
    type="equal", shuffle=True)

headers_nt = headers_nt[3:]
nt = nt[:, 3:]  # remove the first three columns which are not relevant
nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

sim_hnd_cols = [1, 3, 5, 7]
sim_fnd_cols = [0, 2, 4, 6]

sim_hnd_norm = sim_norm[:, sim_hnd_cols]
sim_fnd_norm = sim_norm[:, sim_fnd_cols]

# ----------------------------
# Editable parameters
X = sim_hnd_norm
n_clusters = 4
# ----------------------------

kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)  # Fitting the input data
labels = kmeans.predict(X)  # Getting the cluster labels
centroids = kmeans.cluster_centers_  # Centroid values

print(centroids)  # From sci-kit learn

# ============================
# ANALISYS
# ============================



# ----------------------------
# Select specific rows
selected_rows = np.where(sim_hnd_norm[:, 0] >= 0.8)
# ----------------------------
list_of_selected_labels = list(labels[selected_rows])

for i in range(0, n_clusters):
    print("Cluster ", i, " occurrences: ", list_of_selected_labels.count(i))  # count the occurrences of the cluster 0

print("----------------")
# ----------------------------
# Select specific rows
selected_rows = np.where(sim_hnd_norm[:, 0] <= 0.2)
# ----------------------------

list_of_selected_labels = list(labels[selected_rows])

for i in range(0, n_clusters):
    print("Cluster ", i, " occurrences: ", list_of_selected_labels.count(i))  # count the occurrences of the cluster 0

print("----------------")
# ----------------------------
# Select specific rows
selected_rows = np.where(nt_norm[:, 1] == 1.)  # where %AGGR == 0./0.4/0.7/1.0
# ----------------------------

list_of_selected_labels = list(labels[selected_rows])

for i in range(0, n_clusters):
    print("Cluster ", i, " occurrences: ", list_of_selected_labels.count(i))  # count the occurrences of the cluster 0
print("----------------")
