from get_local_features import get_local_features
from train_codebook import train_codebook
from ressize import resize_image
import pickle
import numpy as np
from params import get_params
import os
import cv2
import time
import warnings
warnings.filterwarnings("ignore")

def codebook(params):

    
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],'train.txt'),'r') as f:
        train_image_list = f.readlines()
    
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'], 'val.txt'),'r') as f:
        val_image_list = f.readlines()
    
     

    descriptors = []
    
    #if params['split'] == 'train':
        
    for img in train_image_list:
        im = cv2.imread(os.path.join(params['root'],params['database'],'train','images',img.rstrip()))
        
        # Resize image
        im = resize_image(params,im)
    
        des=get_local_features(params,im)
        
        if len(descriptors) == 0:
            descriptors = des
        else:
            descriptors = np.vstack((descriptors,des))
            
    print "descriptors train: "
    print np.shape(descriptors)
    for img in val_image_list:
        im = cv2.imread(os.path.join(params['root'],params['database'],'val','images',img.rstrip()))
        
        # Resize image
        im = resize_image(params,im)
    
        des=get_local_features(params,im)
        
        if len(descriptors) == 0:
            descriptors = des
        else:
            descriptors = np.vstack((descriptors,des))

    print "descriptors train + val: "
    print np.shape(descriptors)
    code=train_codebook(params,descriptors)
        
        # Save to disk
    pickle.dump(code,open(os.path.join(params['root'],params['root_save'],
                                        params['codebooks_dir'],'codebook_train_val'
                                        + str(params['descriptor_size']) + "_"
                                        + params['descriptor_type']
                                        + "_" + params['keypoint_type'] + '.cb'),'wb'))
                                        
    
if __name__ == "__main__":

    params = get_params()
    
    print "Creating codebook..."
    t = time.time()
    codebook(params)
    print "Done. Time elapsed:", time.time() - t

