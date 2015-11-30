# -*- coding: utf-8 -*-
import cv2
import numpy as np
import params 

def get_local_features(params, img):
 
    #Leemos una imagen
    image = cv2.imread(img)
    
    #Hacemos más pequeña la imagen
    res = cv2.resize(image ,(500,500))
    
    # Iniciamos el detector SIFT
    sift = cv2.SIFT()
    
    #Guardamos los puntos de interés y sus respectivos descriptores
    kp, des=sift.detectAndCompute(res,None,100)
    
    #Devolvemos los descriptores para una imagen
    return kp,des

  


   
    
   
    
    
    
    
