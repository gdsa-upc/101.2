from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
 
print len(y_true),len(y_pred)

 #precision &recall
 
precision_score(y_true, y_pred, average='weighted')
recall_score(y_true, y_pred, average='weighted') 