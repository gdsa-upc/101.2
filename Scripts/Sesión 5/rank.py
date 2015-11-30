from param import get_params
from feature_extraction import feature_extraction
from sklearn import metrics
import numpy
import os


def rank():
    
 param=get_params()   
 distancia=dict()
 
 Bow_Train,Bow_Val =feature_extraction(param)

 Names= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\TerrassaBuildings900\val\images')
 array=numpy.arange(len(Names))
 for dex in range(len(array)):
  writePath='C:\Users\Bananenbaum\Desktop\GDSA\projecto\Session4\Rankings\{}'.format(array[dex])  
  current=open(writePath,'r') 
  
      
  for bowVal in Bow_Val:
   for bowTrain in Bow_Train:
  
    distancia[bowTrain]=metrics.pairwise.pairwise_distance(bowVal,bowTrain,metric='euclidean',n_jobs=1)
    Currentsort= sorted(distancia)
 
    for linea in  Currentsort:
     current.write(linea+'\n')
    