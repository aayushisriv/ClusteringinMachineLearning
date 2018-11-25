import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
dataset = pd.read_csv('C:/clusterincluster.csv')
X = dataset.values[:,1:]
y = dataset.values[:,2:]
import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=2, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)
for k in range(2, 10):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    labels = kmeans_model.labels_
    print (k, metrics.calinski_harabaz_score(X, labels))
