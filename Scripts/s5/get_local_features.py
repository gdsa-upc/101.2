# -*- coding: utf-8 -*-
import cv2
#from ressize import resize_image

def get_local_features(params, image):
 
    #Leemos una imagen
    #image = cv2.imread(img)
    
    #Hacemos más pequeña la imagen
    #image = resize_image(params,image)
    
    # Iniciamos el detector SIFT
    sift = cv2.SIFT()
    
    #Guardamos los puntos de interés y sus respectivos descriptores
    kp, des=sift.detectAndCompute(image,None)
    
    #Devolvemos los descriptores para una imagen
    return des
    
    