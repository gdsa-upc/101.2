
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

def get_assignments(codebook,des):
    
    # Normalizamos los descriptores
    descriptors = normalize(des, 'l2', 1, True)
    
    # Calculamos las asignaciones para cada descriptor
    assignments = codebook.predict(descriptors)
    #[assig,distance] = vq(descriptors,codebook)
    
    return assignments

