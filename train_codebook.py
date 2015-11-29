# -*- coding: utf-8 -*-
from scipy.cluster.vq import kmeans, vq
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from get_local_features import get_local_features

def train_codebook(des):
    
    # Normalizamos los descriptores
    descriptors = preprocessing.normalize(des, 'l2', 1, True)
    
    #The number of centroids to generate.
    n_clusters = 4
    
    #A k by N array of k centroids.
    #The distortion between the observations passed and the centroids generated.
    [centroides,distorsion] = kmeans(descriptors, n_clusters, iter=20, thresh=1e-05, check_finite=True)
    
    return centroides
    
"""

desc = get_local_features(2,"testimage.jpg")
centroide = train_codebook(2,desc)

plt.scatter(desc[:,0],desc[:,1]),plt.scatter(centroide[:,0],centroide[:,1], color ='r'),plt.show()

"""