# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries: Load essential libraries for data handling, modeling, and evaluation.
2. Load the Iris dataset: Use load_iris() to load the dataset into memory.
3. Convert to DataFrame: Create a pandas DataFrame with feature data and target labels.
4. Define features and target: Separate the dataset into features (x) and target (y).
5. Split data: Divide the data into training and testing sets using train_test_split
6. Initialize SGD Classifier: Create an SGDClassifier with specified hyperparameters (max_iter tol).
7. Train the model: Fit the classifier on the training data (X_train, y_train).
8. Make predictions: Predict outcomes for the test set (x_test).
9. Evaluate accuracy: Compute model accuracy using accuracy_score
10. Generate confusion matrix: Compute a confusion matrix to evaluate prediction performance.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Naveenaa A K
RegisterNumber:  21222230094

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']= iris.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train,y_train)

y_pred =sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test, y_pred)
print("confusion Matrix:")
print(cm)
*/
```

## Output:


![image](https://github.com/user-attachments/assets/8c79d7c4-2fb3-4a69-b780-37f031a5a85b)

![image](https://github.com/user-attachments/assets/9b4cb784-9c5e-4bac-9955-4bfdfbb21d29)

![image](https://github.com/user-attachments/assets/eae86d88-166c-4ee8-9369-884d697bf3da)

![image](https://github.com/user-attachments/assets/6179e172-d5ad-4181-b465-36229bd410f8)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
