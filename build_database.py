import os

#create a file where all imagenames with the ID are saved
file = open('C:\Users\Albert\Documents\UNI\Q-5\GDSA\Projecte\Valoration.txt','w')


#read filenames from the directory
filename= os.listdir(r'C:\Users\Albert\Documents\UNI\Q-5\GDSA\Projecte\TerrassaBuildings900\val\images')
 
 # write the Index and the filename line by line in the textfile
for index in range(len(filename)):
 file.write('{:d}  {}\n'.format(index, filename[index]))
 

file.close()
