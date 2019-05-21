"""
Prototype of program that reads the file produced by 'transitionNfields' and recovers the arrays of the parameters.
Authors: Daan Verweij, Marc Barroso.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca(x, xs, ncomp, whiten=False):
    start_time = time.time()
    scaler = StandardScaler()
    scaler.fit(x)
    X_scaled = scaler.transform(x)
    pca = PCA(n_components=ncomp, whiten=whiten)
    # fit PCA model to breast cancer data
    pca.fit(X_scaled)
    # transform data onto the first two principal components
    X_pca = pca.transform(X_scaled)
    if ncomp==2:
        plt.scatter(X_pca[np.where(xs==0), 0], X_pca[np.where(xs==0),1], label='unstable points')
        plt.scatter(X_pca[np.where(xs==1), 0], X_pca[np.where(xs==1),1], label='stable points')
    elif ncomp==3:
        plt.scatter(X_pca[np.where(xs==0), 0], X_pca[np.where(xs==0),1], c=X_pca[np.where(xs==0), 2], cmap=plt.cm.get_cmap('RdBu_r'), label='unstable points')
        plt.legend()
        plt.show()
        plt.scatter(X_pca[np.where(xs==1), 0], X_pca[np.where(xs==1),1], c=X_pca[np.where(xs==1), 2], cmap=plt.cm.get_cmap('RdBu_r'), label='stable points')
    plt.legend()
    print(" --- %s seconds ---" % (time.time() - start_time))
    plt.show()

def tsne(x, xs, ncomp):
    start_time = time.time()
    tsne = TSNE(n_components=ncomp, random_state=41, perplexity=80.0, learning_rate = 500.0, n_iter=5000)
    # use fit_transform instead of fit, as TSNE has no transform method
    X_tsne = tsne.fit_transform(x)
    if ncomp==2:
        plt.plot(X_tsne[np.where(xs==0),0], X_tsne[np.where(xs==0),1], 'go', label='unstable points')
        plt.plot(X_tsne[np.where(xs==1),0], X_tsne[np.where(xs==1),1], 'ro', label='stable points')
    elif ncomp==3:
        plt.scatter(X_tsne[np.where(xs==0), 0], X_tsne[np.where(xs==0),1], c=X_tsne[np.where(xs==0), 2], cmap=plt.cm.get_cmap('RdBu_r'), label='unstable points')
        plt.legend()
        plt.show()
        plt.scatter(X_tsne[np.where(xs==1), 0], X_tsne[np.where(xs==1),1], c=X_tsne[np.where(xs==1), 2], cmap=plt.cm.get_cmap('RdBu_r'), label='unstable points')
    plt.legend()
    print(" --- %s seconds ---" % (time.time() - start_time))
    plt.show()

def all_plots(x):
    points_dataframe = pd.DataFrame(x, columns=('l1','log(l2)','l3','gx'))
    grr = pd.plotting.scatter_matrix(points_dataframe, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
    plt.show()

def reading_file(file_name):
    f = open(file_name, 'r')
    f.readline() #We read and disregard the first line, since it only contains the structure of the file
    f1 = f.readlines() #We save the other lines
    x = np.array([[0,0,0,0]])
    xs = np.array([[0]])
    for i in range(len(f1)):
        if i % 2 == 1: #we visit only the odd lines, since we know that here is '-'*90 between them
            tmp = f1[i].split(' ') #we split each line in sections separated by ' '
            x = np.concatenate((x, [[float(tmp[0]), np.log10(-float(tmp[1])), float(tmp[2]), float(tmp[3])]]), axis=0) #and save the numbers in the array
            xs = np.concatenate((xs, [[float(tmp[8])]]), axis=0)
    x = np.delete(x, (0), axis=0)
    xs = np.delete(xs, (0), axis=0)
    f.close()
    return x, xs

if __name__ == '__main__':
    start_time = time.time()
    """
    x, xs = reading_file('2fields.txt')
    xt, xst = reading_file('2fields_minusl2.txt')
    x = np.concatenate((x,xt), axis=0)
    xs = np.concatenate((xs,xst), axis=0)
    """
    x, xs = reading_file('2fields_minusl2.txt')
    #print x[np.where(x[:,1]==np.amax(x[:,1]))] #show me x where the value of the column is equal to the maximum value of that column. It might show more than just one entry
    xstable = x[np.where(xs==1)[0],:]
    xunstable = x[np.where(xs==0)[0],:]
    print xstable.shape
    print xunstable.shape
    print(" --- %s seconds ---" % (time.time() - start_time))
    all_plots(xstable)
    all_plots(xunstable)
    #tsne(x, xs, 2)
    #pca(x, xs, 2)
