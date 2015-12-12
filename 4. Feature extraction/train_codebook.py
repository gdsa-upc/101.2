# -*- coding: utf-8 -*-
#from sklearn.preprocessing import normalize
#from params import get_params
from sklearn.cluster import MiniBatchKMeans

def train_codebook(params,des):
    
    # Normalizamos los descriptores
    #descriptors = normalize(des, 'l2', 1, True)
    
    #A k by N array of k centroids.
    #The distortion between the observations passed and the centroids generated.
    km = MiniBatchKMeans(params['descriptor_size'])
    
    km.fit(des)
    
    return km
    