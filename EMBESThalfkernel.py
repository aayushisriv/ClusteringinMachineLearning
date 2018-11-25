import pandas as pd
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn import datasets
dataset = pd.read_csv('C:/halfkernel.csv')
X = dataset.values[:,1:]
y = dataset.values[:,2:]
import numpy as np
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k).fit(X)
    labels = gmm.predict(X)
    print (k, metrics.calinski_harabaz_score(X, labels))

