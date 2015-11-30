
from scipy.cluster.vq import vq
import sklearn 

def get_assignments(des,codebook):
    
    # Normalizamos los descriptores
    descriptors = sklearn.preprocessing.normalize(des, 'l2', 1, True)
    
    # Calculamos las asignaciones para cada descriptor
    [assig,distance] = vq(descriptors,codebook)
    
    return assig

