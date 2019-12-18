import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statistics import mean

DATA = pd.read_csv("data.csv", delimiter=',', index_col='Object')

coords = DATA.drop('Cluster', axis=1)
kmeans = KMeans(n_clusters=3, init=np.array([[13.5, 8.75], [11.67, 10.33], [9.8, 9.0]]), max_iter=100, n_init=1)
model = kmeans.fit(coords)
print('Clusters:\n', model.labels_.tolist())
alldistances = kmeans.fit_transform(coords)
print(alldistances[0])
print('Average for 0th element: ', round(mean(alldistances[0]), 3))