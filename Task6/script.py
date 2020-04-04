import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import pandas

FILENAME = '94_16.csv'

X = pandas.read_csv(FILENAME, header=None)
print(X.shape)

pca = PCA(n_components = 2, svd_solver='full')
X_transformed = pca.fit(X).transform(X)

# First object coordinates
print('Coordinates of 1st object: ', X_transformed[0])

# TODO: uncomment to see the pic and count number of "clusters"
#plt.plot(X_transformed[:101, 0], X_transformed[:101, 1], 'o', \
#	markerfacecolor='red', markeredgecolor='k', markersize=8)
#plt.show()

# Explained variance
pca = PCA(n_components=10, svd_solver='full')
X_full = pca.fit(X).transform(X)
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)
print('Explained variance for 2 main components: ', explained_variance[2])
# TODO: uncomment to see pic
#plt.plot(np.arange(start=1, stop=11), explained_variance, ls = '-')
#plt.show()
for i in range(len(explained_variance)):
	print('Components: ', i + 1, ' variance: ', explained_variance[i])