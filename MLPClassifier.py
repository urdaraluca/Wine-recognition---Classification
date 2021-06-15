import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 


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

plt.close('all')
sns.pairplot( data,hue='name')


X=data.iloc[:,6:8] 

T=data['name'].replace(classes,[1,2,3])


xTrain, xTest, tTrain, tTest = train_test_split(X,T, test_size = 0.3)

net = MLPClassifier(solver='sgd', alpha=1e-5, verbose=1,max_iter=5000,
hidden_layer_sizes=(5, 3), random_state=1)
net.fit(xTrain, tTrain)
yTest = net.predict(xTest)
print('The accuracy is:',accuracy_score(tTest,yTest)) 
# accuracy_score(y_true, y_pred)
print('Confusion Matrix is: ')
print(confusion_matrix(tTest,yTest)) 
# confusion_matrix(y_true, y_pred) 

plt.figure()
loss_values = net.loss_curve_
plt.plot(loss_values)
plt.title('Loss function')