import os
import pickle
import numpy as np
from params import get_params
from sklearn.metrics.pairwise import pairwise_distances
from kaggle_scripts import save_ranking_file
import pandas as pd

def rank(params):
    
    # Load train and validation feature dictionaries
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))

    train_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             'train' + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))
    
    annotation_val = pd.read_csv(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'), sep='\t', header = 0)
    out = open(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'], 'rank.csv'),'w')
    
    
    # For each image id in the validation set
    for val_id in val_features.keys():

        # Get its feature
        bow_feats = val_features[val_id]

        # The ranking is composed with the ids of all training images
        ranking = train_features.keys()
        
        X = np.array(train_features.values())

        # The .squeeze() method reduces the dimensions of an array to the minimum. E.g. if we have a numpy array of shape (400,1,100) it will transform it to (400,100)
        distances = pairwise_distances(bow_feats,X.squeeze())


        # Sort the ranking according to the distances. We convert 'ranking' to numpy.array to sort it, and then back to list (although we could leave it as numpy array).
        ranking = list(np.array(ranking)[np.argsort(distances.squeeze())])
        
        # Save to text file
        outfile = open(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],val_id.split('.')[0] + '.txt'),'w')
        
        for item in ranking:

            outfile.write(item.split('.')[0] + '\n')
        
        outfile.close()
    
    out.write("Query,RetrievedDocuments\n")
    for val_id in os.listdir(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'])):
        
        query_class = list(annotation_val.loc[annotation_val['ImageID'] == val_id.split('.')[0]]['ClassID'])[0]     
        rank = pd.read_csv(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],val_id.split('.')[0] + '.txt'),header= None)
        
        if not query_class == "desconegut":
            
            save_ranking_file(out,val_id,rank)
        

if __name__ == "__main__":
    
    params = get_params()
    rank(params)