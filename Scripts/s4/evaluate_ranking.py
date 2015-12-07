import os
import numpy
import string

#save the names of the files in the Ranking archive 

rankingNames= os.listdir(r'C:\Users\Bananenbaum\Desktop\GDSA\projecto\Ranking_val')

#the ranking lists are opend secuentionaly

array=numpy.arange(len(rankingNames))

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

 array[dex]=correct


