#EX06 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.


## Program:
```
Developed by: vidhyasri.k

RegisterNumber:  212222230170
import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
Data.head():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/bb44c37c-c708-4c2e-96fe-21e368c44b70)

Data.info():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/5792b208-a8ba-4e47-9317-9b95d59473e8)

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/2683057e-da6c-46fd-b428-aa7d44f649c3)

Data Value Counts():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/0132f4e8-ce55-4a0b-96b4-5fd8f47fd1b9)

Data.head() for salary:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/f0e30abe-85e8-46e3-898f-8b9c638aa02e)

x.head():

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/a8d7bcda-7e8f-4744-9ea4-302a8a99deb9)

Accuracy Value:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/612179ac-87dd-459d-b2e6-319ba726eb06)

Data Prediction:

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477817/5a10fad1-03d1-4ae5-93cb-3a513a0f2845)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
