# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset using chardet
2. Get dataset info and check for null values
3. Assign x and y values and split the dataset into training and testing sets
4. Import CountVectorizer and transform x_train,x_test as vectors
5. Import SVC and fit it to dataset
6. Find y predict and accuracy 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Manjupriya P
RegisterNumber: 212220220024 
*/
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:
RESULT OUTPUT:

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/ab09c23b-4957-4bae-8266-b840624b6793)

DATA.HEAD():

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/d3d4e110-ba6a-42ec-a116-57f2be56e777)

DATA.INFO()

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/f2131c53-c814-4bbf-88d3-87854be495cd)

DATA.ISNULL().SUM():

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/85ddbfe5-647d-40a1-adcd-eabc473c3ff2)

Y_PREDICTION VALUE:

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/e4c7070f-01a9-4837-89bd-ac243cb3d655)

ACCURACY VALUE:

![image](https://github.com/Manjupriya1207/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113583090/74e4e986-5365-425d-9d19-89d5ea0a5348)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
