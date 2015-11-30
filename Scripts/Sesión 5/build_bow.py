import numpy as np
from sklearn import preprocessing


def build_bow(assignment,codebook):

    Bow_vector=dict()
    
    for index in assignment:
        temp = dict()
    
        for one in assignment[index]:
            two =one+1
            temp[one] = temp[two]
    
        for second in temp:
            temp[second] = preprocessing.normalize(second, 'l2', 1, True)
            Bow_vector[index]=temp
    
    return Bow_vector