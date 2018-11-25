import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics 
from sklearn.mixture import GaussianMixture

# Column names to the dataset
names = ['Class', 'x1', 'x2']

# Read dataset in csv format to pandas dataframe
df = pd.read_csv('C:/twospirals.csv')

#Class label column is not used for this program as we need to ignored the label e.g: 'X'
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values 
gmm = GaussianMixture(n_components=2)

z=gmm.fit(X).predict(X)
centroids=gmm.means_

for a,b,c,d in zip(X[:,0],X[:,1],z,y):
    if c==1:
        if d==1:
            plt.scatter(a,b,facecolors='none', edgecolors='red')
        else:
            plt.scatter(a,b,facecolors='none', edgecolors='green')
        
    else:
        if d==1:
            plt.scatter(a,b,c='red',marker='+')
        else:
            plt.scatter(a,b,c='green',marker='+')
plt.scatter(centroids[:,0],centroids[:,1],c='#000000',marker='x')    
plt.show()

