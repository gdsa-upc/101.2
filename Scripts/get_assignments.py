
from scipy.cluster.vq import vq
from sklearn.preprocessing import normalize

def get_assignments(codebook,des):
    
    # Normalizamos los descriptores
    descriptors = normalize(des, 'l2', 1, True)
    
    # Calculamos las asignaciones para cada descriptor
    [assig,distance] = vq(descriptors,codebook)
    
    return assig

