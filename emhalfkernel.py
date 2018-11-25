import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn import mixture
#%matplotlib inline
#import numpy as np  
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture


#import seaborn as sns

# Assign colum names to the dataset
names = ['Class', 'x1', 'x2']

# Read dataset to pandas dataframe
df = pd.read_csv('C:\halfkernel.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values 

gMixture = GaussianMixture(n_components=2)  
z=gMixture.fit(X).predict(X);
for a,b,c,d in zip(X[:,0],X[:,1],z,y):
    
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
#plt.scatter(centroids[:,0],centroids[:,1],c='#000000',marker='x')    
plt.show()
