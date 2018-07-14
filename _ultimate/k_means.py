import numpy as np
import matplotlib.pyplot as plt

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd

import _ultimate.manage_data.data_normalize as dn
from sklearn.cluster import KMeans

plotly.tools.set_credentials_file(username='rick.allin', api_key='GQ0USxdKxs8K92rKjGG9')

nt, avg_layers, avg_chxrounds, sim, nt_norm, avg_layers_norm, avg_chxrounds_norm, sim_norm, headers_nt, headers_avg_layers, headers_avg_chxrounds, headers_sim = dn.load_normalized_data(
    type="equal", shuffle=True)

headers_nt = headers_nt[3:]
nt = nt[:, 3:]  # remove the first three columns which are not relevant
nt_norm = nt_norm[:, 3:]  # remove the first three columns which are not relevant

sim_hnd_cols = [1, 3, 5, 7]
sim_fnd_cols = [0, 2, 4, 6]

hnd_headers = headers_sim[sim_hnd_cols]
fnd_headers = headers_sim[sim_fnd_cols]

sim_hnd_norm = sim_norm[:, sim_hnd_cols]
sim_fnd_norm = sim_norm[:, sim_fnd_cols]

all_data_hnd_norm = np.append(nt_norm, sim_hnd_norm, axis=1)
all_data_fnd_norm = np.append(nt_norm, sim_fnd_norm, axis=1)

hnd_protocols_names = headers_sim[sim_hnd_cols]
fnd_protocols_names = headers_sim[sim_fnd_cols]

best_hnd_protocols = np.argmax(sim_norm[:, sim_hnd_cols], axis=1)  # index of the most efficient protocol
best_fnd_protocols = np.argmax(sim_norm[:, sim_fnd_cols], axis=1)

# ----------------------------
# Editable parameters
X = sim_hnd_norm
n_clusters = 3
# ----------------------------

kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)  # Fitting the input data
labels = kmeans.predict(X)  # Getting the cluster labels
centroids = kmeans.cluster_centers_  # Centroid values

# print(centroids)  # From sci-kit learn

# ============================
# ANALISYS
# ============================

# show the best efficiency cluster for each protocol
print("Efficiency >= 0.7")
for prot_index in range(0, X.shape[1]):
    # ----------------------------
    # Select specific rows
    selected_rows = np.where(sim_hnd_norm[:, prot_index] >= 0.7)
    # ----------------------------
    list_of_selected_labels = list(labels[selected_rows])

    for i in range(0, n_clusters):
        print(hnd_headers[prot_index], ") Cluster ", i, " occurrences: ",
              list_of_selected_labels.count(i))  # count the occurrences of the cluster 0

print("----------------")
print("Efficiency <= 0.1")
for prot_index in range(0, X.shape[1]):
    # ----------------------------
    # Select specific rows
    selected_rows = np.where(sim_hnd_norm[:, prot_index] <= 0.1)
    # ----------------------------
    list_of_selected_labels = list(labels[selected_rows])

    for i in range(0, n_clusters):
        print(hnd_headers[prot_index], ") Cluster ", i, " occurrences: ",
              list_of_selected_labels.count(i))  # count the occurrences of the cluster 0

print("----------------")
print("%AGGR = 0. or .4")
# ----------------------------
# Select specific rows

selected_rows = np.where(nt[:, 1] == 0.) or np.where(nt[:, 1] == .4)  # where %AGGR == 0./0.4/0.7/1.0
# ----------------------------

list_of_selected_labels = list(labels[selected_rows])

for i in range(0, n_clusters):
    print("Cluster ", i, " occurrences: ", list_of_selected_labels.count(i))  # count the occurrences of the cluster 0
print("----------------")

"""
https://plot.ly/python/parallel-coordinates-plot/
"""
all_data_to_print = np.empty((0, 7))

for c_id in range(0,3): # iterate the cluster ids
    cluster_id = c_id
    selected_rows_index = np.where(labels == cluster_id)
    data_to_print = nt_norm[selected_rows_index]
    # print(data_to_print[:4])

    cluster_id_extra_column = np.zeros((len(data_to_print), 1))
    cluster_id_extra_column = cluster_id_extra_column + cluster_id
    data_to_print = np.append(data_to_print, cluster_id_extra_column, axis=1)
    # print(data_to_print[:4])
    all_data_to_print = np.append(all_data_to_print, data_to_print, axis=0)

headers_nt = np.append(headers_nt, ['CLUSTER ID'], axis=0)
headers_nt = headers_nt.T

# df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/iris.csv")

print(all_data_to_print.shape)
df = pd.DataFrame(all_data_to_print, columns=headers_nt)
print(df[2500:2504])

data = [
    go.Parcoords(
        line=dict(color=df["CLUSTER ID"], colorscale=[[.2, '#0000FF'], [.2, '#FF0000'], [.2, '#00FF00']]),
        dimensions=list([
            dict(range=[0, 1],
                 label='R0', values=df["R0"]),
            dict(range=[0, 1],
                 label='%AGGR', values=df["%AGGR"]),
            dict(range=[0, 1],
                 label='HET', values=df["HET"]),
            dict(range=[0, 1],
                 label='HOM ENERGY', values=df["HOM ENERGY"]),
            dict(range=[0, 1],
                 label='HOM RATE', values=df["HOM RATE"]),
            dict(range=[0, 1],
                 label='DENSITY', values=df["DENSITY"])
        ])
    )
]

layout = go.Layout(
    plot_bgcolor='#E5E5E5',
    paper_bgcolor='#E5E5E5'
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='parcoords-basic')
