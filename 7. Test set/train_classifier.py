# -*- coding: utf-8 -*-
import os
import pickle
from sklearn import svm
from params import get_params
import numpy as np
#from sklearn.grid_search import GridSearchCV

def train_classifier(params):
    params['split']='train'
   
    train_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],params['split'] + '_' + str(params['descriptor_size']) + '_' + params['descriptor_type'] + '_' + params['keypoint_type'] + '.p'),'rb'))
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],'val' + '_' + str(params['descriptor_size']) + '_' + params['descriptor_type'] + '_' + params['keypoint_type'] + '.p'),'rb'))
    #annotation= pd.read_csv(os.path.join(params['root'],params['database'],'train','annotation.txt'), sep='\t', skiprows = 1, header = 0)
    with open(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'),'r') as a:
        train_annotation = a.readlines()
    with open(os.path.join(params['root'],params['database'],'val','annotation.txt'),'r') as an:
        val_annotation = an.readlines()
    
    feats = [] # Llista d'arrays de features
    ids = [] # Llista d'identificadors
    class_annot = [] # Llista de classes
    
    #for train_id in train_features.keys(): 
    for key,values in train_features.iteritems():
        feats.append(values)
        ids.append(key.split('.')[0])
    
    for key,values in val_features.iteritems():
        feats.append(values)
        ids.append(key.split('.')[0])
    # print feats ---> feats a saco paco...
    #print 'Dimensión feats:' 
    #print np.shape(feats)
    print 'Dimensión feats con np.squeeze:' 
    #---> (450L, 128L)
    print np.shape(np.squeeze(feats))
    

    feats = np.squeeze(feats) # Reducimos la dimensión del array. Antes: (450L, 1L, 128L), ahora: (450L, 128L)
        
        
    for idd in ids:
        #print idd
        
        for annot in  train_annotation:
                #print 'annot_ID   ---->   ' + annot.split()[0] 
                #print 'clase_annot   ---->   ' + annot.split()[1] 
                
                x = annot.split()[1]
                if annot.split()[0] == idd:
                    class_annot.append(x)
                    
        for annot in  val_annotation:
                #print 'annot_ID   ---->   ' + annot.split()[0] 
                #print 'clase_annot   ---->   ' + annot.split()[1] 
                
                x = annot.split()[1]
                if annot.split()[0] == idd:
                    class_annot.append(x)
        

    d = 0 
    cl = 0 
    for clas in class_annot:
        if clas == 'desconegut':
            d = d + 1
        else :
            cl = cl + 1
            

    total_n_samples = len(ids)
    n_classes = len(np.unique(class_annot))
    cl = cl/(n_classes-1)
    
    print 'Total n samples: ' 
    print total_n_samples
    print 'Número de clases: ' 
    print n_classes
    print cl 
    print d                      
                                              
    classes = {}

    for c in set(class_annot):
        if c == 'desconegut':
            #cla = float(total_n_samples / (n_classes * d))
            classes[c] = 0.23076923
        else:
            cla = total_n_samples / (n_classes * cl)
            classes[c] = cla
    
    '''
    y = np.arange(1,500)
    lista = []
    for yy in y:
        lista.append(yy)
    #print lista       
    svr = svm.SVC()        
    parameters = {'kernel':('linear','rbf'), 'C':[1,10,100,1000,10000], 'gamma':[0.1,0.01,0.001,0.0001,0.00001]}
    a = GridSearchCV(svr,parameters)
    a.fit(feats,class_annot)
    print a.best_params_
    '''        

    sv = svm.SVC(C=10, kernel='linear', degree=3, gamma=0.1, coef0=0.0, shrinking=True, 
    probability=False, tol=0.001, cache_size=200, class_weight = classes, verbose=False, max_iter=-1, 
    decision_function_shape=None, random_state=None)
    
    
    #print class_annot
    #print np.shape(class_annot)
    print 'Peso de cada clase: ' 
    print classes
    
    svm_dict = sv.fit(feats,class_annot)
    
    # Save dictionary to disk with unique name                    
    save_file = os.path.join(params['root'],params['root_save'],params['classification_dir'],
                            params['descriptor_type'], params['split'] + '_classification' + '.p')

    pickle.dump(svm_dict,open(save_file,'wb'))
    
    
    
    
if __name__ == "__main__":
    
    params = get_params()
    
    train_classifier(params)


