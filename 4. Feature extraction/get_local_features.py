# -*- coding: utf-8 -*-
import cv2
from rootsift import RootSIFT
from params import get_params
#from ressize import resize_image

def get_local_features(params, image):

    # detect Difference of Gaussian keypoints in the image
    detector = cv2.FeatureDetector_create(params['keypoint_type'])
    kps = detector.detect(image)
    
    # extract RootSIFT descriptors
    rs = RootSIFT()
    (kps, descs) = rs.compute(image, kps)

    
    #Devolvemos los descriptores para una imagen
    return descs


    