import numpy as np
from sklearn.preprocessing import normalize


def build_bow(assignment,codebook):

    tamany_bw = np.shape(codebook[0])
    bow = np.zeros(tamany_bw)
    
    for index in assignment:
        bow[index] += 1
    
    bow = normalize(bow)
        
    return bow