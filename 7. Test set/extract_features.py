from get_local_features import get_local_features
from get_assignments import get_assignments
from build_bow import build_bow
from ressize import resize_image
import pickle
from params import get_params
import os
import cv2
import time
import warnings
warnings.filterwarnings("ignore")


def extract_features(params):

    feats_dic = {}
    
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()


    # Get trained codebook
    code = pickle.load(open(os.path.join(params['root'],params['root_save'],
                                     params['codebooks_dir'],'codebook_train_val_'
                                     + str(params['descriptor_size']) + "_"
                                     + params['descriptor_type']
                                     + "_" + params['keypoint_type'] + '.cb'),'rb'))
        
    for img in image_list:
        
        im = cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',img.rstrip()))
        im = resize_image(params,im)
        des = get_local_features(params,im)
        assign = get_assignments(code,des)
        feats_dic[img] = build_bow(assign,code)
        
    # Save dictionary to disk with unique name
    save_file = os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p')

    pickle.dump(feats_dic,open(save_file,'wb'))
    
    
    
if __name__ == "__main__":

    params = get_params()

    params['descriptor_size'] = 512
    
    params['split'] = 'test'
    
    print "Storing bow features for testing set..."
    t = time.time()
    extract_features(params)
    print "Done. Time elapsed:", time.time() - t