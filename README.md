# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries (e.g., pandas, numpy,matplotlib).

2.Load the dataset and then split the dataset into training and testing sets using sklearn library.

3.Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).

4.Use the trained model to predict marks based on study hours in the test dataset.

5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SWATHI D
RegisterNumber: 212222230154

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

## df.head()
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/d64b9d18-7b39-422d-8967-cf750e4ad4c1)

## Array value of X
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/9962ab52-f6b7-4143-95fd-9479f34c73e3)

## Array value of Y
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/728df3ad-a1d9-4945-a6f2-ca885220e96d)

## Values of y prediction
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/ec9cd745-f12e-4044-beaa-8f78d48a7f11)

## Training set
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/f854ae71-d3ac-4e40-bd14-25a20359553f)

## Values of MSE,MAE and RMSE
![image](https://github.com/swathi22003343/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120440439/b4e0fc8f-0e28-46b8-a346-4df723954496)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
