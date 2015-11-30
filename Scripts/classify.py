import random

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