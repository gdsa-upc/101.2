# -*- coding: utf-8 -*-
import os
import pickle
from params import get_params
import numpy as np

def classifier(params):
    
    # Cargamos el diccionario val_features.
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],
                params['feats_dir'],params['split'] + '_' + str(params['descriptor_size']) 
                + '_' + params['descriptor_type'] + '_' + params['keypoint_type'] + '.p'),'rb'))
    
    # Cargamos el modelo de entrenamiento, hecho en train_classifier.
    train_model = pickle.load(open(os.path.join(params['root'],params['root_save'],
                    params['classification_dir'],params['descriptor_type'],'train_classification.p'),'rb'))
    
    # Fichero de salida.
    outfile = open(os.path.join(params['root'],params['root_save'],params['classification_dir'],
                    params['descriptor_type'],params['split'] + '_classification.txt'),'w')
    
    ids = [] # Lista de identificadores.
    val_feats = [] # Lista de features de validación.
                                    
    for key,values in val_features.iteritems():
        
        val_feats.append(values) # Rellenamos la lista con BoWs.
        ids.append(key.split('.')[0]) # Con split(.)[0] cogemos solamente el identificador sin '.jpg'.
        
    val_feats = np.squeeze(val_feats) # Reducimos la dimensión del array. Antes: (450L, 1L, 128L), ahora: (450L, 128L).
                   
    prediction_labels = train_model.predict(val_feats) # Hacemos la predicción de val_features respecto al train_model.


    # En este for incluyo los ids en una columna y las clases en otra, segun el predict.
    i = 0           
    for label in prediction_labels:
        
        outfile.write( ids[i] + '\t' + label + '\n') # Rellenamos el fichero con prediction names y prediction labels.
        i = i + 1
        
    outfile.close()
    
    
    
if __name__ == "__main__":
    
    params = get_params()
    
    classifier(params)


