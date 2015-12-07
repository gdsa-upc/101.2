# -*- coding: utf-8 -*-
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import normalize
from params import get_params


def train_codebook(params,des):
    
    # Normalizamos los descriptores
    descriptors = normalize(des, 'l2', 1, True)
    
    #A k by N array of k centroids.
    #The distortion between the observations passed and the centroids generated.
    [codebook,distorsion] = kmeans(descriptors, params['descriptor_size'], iter=20, thresh=1e-05, check_finite=True)
    
    return codebook
    