import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory
from sompy.visualization.mapviewsompy.vi import View2D
from sompy.visualization.bmuhits import BmuHitsView
from  sompy.visualization.hitmapsompy.vi  import HitMapView
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
descr = data.DESCR
names = fetch_california_housing().feature_names + ["HouseValue"]

data = np.column_stack([data.data, data.target])
print(descr)
print("FEATURES: ", ", ".join(names))

# msz = calculate_msz(data)#msz =
sm = SOMFactory().build(data, normalization='var', initialization='random', component_names=names)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

view2D = View2D(10, 10, "rand data", text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

vhts = BmuHitsView(4, 4, "Hits Map", text_size=12)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)


sm.cluster(4)
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)
