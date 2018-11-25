import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics
#%matplotlib inline
#import numpy as np  
from sklearn.cluster import KMeans 
#import seaborn as sns


# Assign colum names to the dataset
names = ['Class', 'x1', 'x2']

# Read dataset to pandas dataframe
df = pd.read_csv( "C:\halfkernel.csv")
#print(df.isnull().sum())
#print(df.head())
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values 
#plt.scatter(X[:,0],X[:,1], label='True Position')

kmeans = KMeans(n_clusters=2)  
kmeans.fit(X) 

centroids=kmeans.cluster_centers_
#print(kmeans.labels_)
print(kmeans.cluster_centers_)
#print(y)
z=kmeans.labels_ 
#print(metrics.davies_bouldin_score(X, z))
#g=sns.lmplot(x="x1",y="x2",data=df,fit_reg=False,hue="Class")
#sns.lmplot(x="x1",y="x2",data=df,fit_reg=False,hue=z,markers=['o','v'])
#plt.scatter(X[:,0],X[:,1],c=y, cmap='rainbow')
#plt.scatter(X[:,0],X[:,1],marker='v')
for a,b,c,d in zip(X[:,0],X[:,1],z,y):
    #print(a,b)
    if c==1:
        if d==1:
            plt.scatter(a,b,facecolors='none', edgecolors='r')
        else:
            plt.scatter(a,b,facecolors='none', edgecolors='b')
        
    else:
        if d==1:
            plt.scatter(a,b,c='r',marker='+')
        else:
            plt.scatter(a,b,c='b',marker='+')
plt.scatter(centroids[:,0],centroids[:,1],c='#000000',marker='x')    
plt.show()
