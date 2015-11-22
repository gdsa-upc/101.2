# -*- coding: utf-8 -*-

import os
import random
import numpy
import string
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


                #Eins
#create a file where all imagenames with the ID are saved

file = open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Valeration.txt','w')


#read filenames from the directory

filename= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\TerrassaBuildings900\val\images')
 
 # write the Index and the filename line by line in the textfile
 
for index in range(len(filename)):
 file.write('{:d}  {}\n'.format(index, filename[index]))
 

file.close()

#create a file where all imagenames with the ID are saved

file = open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Train.txt','w')


#read filenames from the directory

filename= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\TerrassaBuildings900\train\images')
 
 # write the Index and the filename line by line in the textfile
 
for index in range(len(filename)):
 file.write('{:d}  {}\n'.format(index, filename[index]))
 

file.close()

                #Zwei
#read the file that was produced in the step bevor

file = open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Valeration.txt','r')

#create a new directory

Valdir=dict()

#read the file line by line

for line in file.readlines():
    
#only use the first colum with the indexes

 index = line.split()[0]
 x =random.randint(100,900)
 
#safe the indexes of the images together with the 'feature'(random numer)

 Valdir[index] = x
 

file.close()


#read the file that was produced in the step bevor

file = open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Train.txt','r')

#create a new directory

Traindir=dict()

#read the file line by line

for line in file.readlines():
    
#only use the first colum with the indexes

 indexT = line.split()[0]
 z =random.randint(100,900)
 
#safe the indexes of the images together with the 'feature'(random numer)

 Traindir[indexT] = z
 
file.close()

                #Drei

#get the features from the directory 
#that was made in the step before

for feature in Valdir.itervalues():

    #create new textfiles with the feature of the image as name to safe the rankinglist
    
 path= 'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Ranking_val\ {:d}'.format(feature)
 path2= path+'.txt'

    #open the textfile to write
    
 newFile=open(path2,'w')
 
    #read the features from the traindirectory 
    #to compare to tthe current feature
    
 for feature_train in Traindir.itervalues():
     
#write random ranking to the file

  number =random.randint(100,900)
  newFile.write('{} {:d}\n'.format(feature_train, number))

 newFile.close()
 
                #Vier
#create new textfile to safe the annotationlist

newFile2=open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Annotation_list.txt','w')

cathegories=['catedral','desconegut','la_enginyeria','estacio_nord','dona_treballadora','mnactec',
'castell_cartoixa','ajuntament','masia_freixa','teatre_principal','mercat_independencia',
'farmacia_albinyana','societat_general']

#get the features from the directory 
#that was made before

for feature_val in Valdir.itervalues():
    
#write feature with random annotation to the file

 number2 =random.randint(0,12)
 newFile2.write('{} {}\n'.format(feature_val, cathegories[number2]))
newFile2.close()

                #Fünf
#save the names of the files in the Ranking archive 

rankingNames= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Ranking_val')

#the ranking lists are opend secuentionaly

array=[]

for dex in range(len(rankingNames)):
 writePath='C:\Users\Bananenbaum\Desktop\GDSA\projecto\Ranking_val\{}'.format(rankingNames[dex])  
 current=open(writePath,'r')    
 
#initiate a variable to count the correct file number

 correct=0
 x=string.split(rankingNames[dex],'.',1)
 x2=int(''.join(x[0]))
 
#compare if the feature of every picture in the ranking list is equal to the filename

 for line in current.readlines():
  picture_feature= line.split()[0]
  picture_feature2=int(picture_feature)
  if picture_feature2==x2: 
    correct+=1
    
#safe the number of correct files in an array

 array.append(correct)

                #Sechs
#1d array-like, or label indicator array / sparse matrix
#Ground truth (correct) target values.

ground =open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\TerrassaBuildings900\Val\Val_annotation.txt','r')
y_true= []

#Estimated targets as returned by a classifier.

targets =open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Annotation_list.txt','r')
y_pred= []

ground.readline()
for line in ground.readlines():
    
#only use the second column

 currentClass = line.split()[1]
 y_true.append(currentClass)

for line in targets.readlines():
 currentTrue = line.split()[1]
 y_pred.append(currentTrue) 
 

 #precision &recall
 
print 'the precision score is: ' + str(precision_score(y_true, y_pred, average='weighted'))
print 'the recall score is: ' + str(recall_score(y_true, y_pred, average='weighted'))
