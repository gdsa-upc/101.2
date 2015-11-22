import random

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
  print feature_train
  
#write random ranking to the file

  number =random.randint(100,900)
  newFile.write('{} {:d}\n'.format(feature_train, number))

 newFile.close()
 
