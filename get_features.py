#random is a function to produce random numbers in a specific range

import random

#read the file that was produced in the step bevor

file = open('C:\Users\Bananenbaum\Desktop\GDSA\projecto\Valeration.txt','r')

#create a new directory

newdir=dict()

#read the file line by line

for line in file.readlines():
    
#only use the first colum with the indexes

 index = line.split()[0]
 x =random.randint(100,900)
 
#safe the indexes of the images together with the 'feature'(random numer)

 newdir[index] = x

file.close()

for i in range(0,10):
 a=str(i)
 out='ID: '+a+'  feature: '+str(newdir[a])
 print out