from get_local_features import get_local_features
from get_assignments import get_assignments
from train_codebook import train_codebook
from Bow_vector import build_bow

import os
##import cv2

def feature_extraction(param):
    Train_dic =dict()
    Val_dic=dict()
   
    Train_files= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\TerrassaBuildings900\Train\images')
    Val_files= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\TerrassaBuildings900\Val\images')
    
    for index in Train_files:
       
        des=get_local_features(param,'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\TerrassaBuildings900\Train\images\{}'.format(index))
        print des
        code=train_codebook(param,des)
        T_assign=get_assignments(code,des)
        T_Bow=build_bow(T_assign,code)
        Train_dic[index]=T_Bow
    
    for index in Val_files:
    
        des=get_local_features(param,'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\TerrassaBuildings900\Val\images\{}'.format(index))
        code=train_codebook(param,des)
        V_assign=get_assignments(code,des)
        V_Bow=build_bow(V_assign,code)
        Val_dic[index]=V_Bow
        
    return Train_dic, Val_dic