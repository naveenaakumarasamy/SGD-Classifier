# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: Naveenaa A K
RegisterNumber:  212222230094

```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
df = pd.DataFrame(data = iris.data , columns = iris.feature_names)
df['target'] = iris.target
print(df.head())
x = df.drop('target',axis=1)
y = df['target']
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter = 1000 , tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.3f}")
cm = confusion_matrix(y_test,y_pred)
print("confusion Matrix:")
print(cm)
```
## Output:

![image](https://github.com/user-attachments/assets/4846dd24-1c81-4550-90ee-e12652917579)
![image](https://github.com/user-attachments/assets/ed34c96c-c7f2-464c-9eb5-d0a72e4733cb)
![image](https://github.com/user-attachments/assets/5a6086c9-e1b3-4a06-b3aa-65a99039e787)
![image](https://github.com/user-attachments/assets/19629795-bf68-4cf1-abcc-214cd492a08b)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
