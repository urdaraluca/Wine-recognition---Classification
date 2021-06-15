import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_csv('wine.data', header=None)
# data = pd.read_csv('wine.names')

data.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]

classes=['Wine 1','Wine 2','Wine 3']

print('\n')
#pandas - print first three instances
print(data.head(3)) #check out the data

print('\n')
#pandas - print statistical data
print(data.describe())

wine = datasets.load_wine()
x = wine.data
y = wine.target

#  choose 2 features
X=x[:,6:8]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)


classification=svm.SVC(kernel='linear')  
classification.fit(x_train,y_train)
linear = classification.predict(x_test)
print(linear)



print(confusion_matrix(y_test,linear))

target=wine.target_names
print(classification_report(y_test,linear,target_names=target))
             
