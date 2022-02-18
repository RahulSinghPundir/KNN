import numpy as np
import pandas as pd

# Reading the csv data from github repo.
df=pd.read_csv("K_Neareast_Neighbour/teleCust1000t.csv")

# Analyising the features of our data that which our output is dependent.
df.head()

# Splitting the data in two half dependent and independent variable
X=df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']]
Y=df['custcat']
print(X.head())
print(Y.head())

# Looking into our output classification that how many classification (different values) we have in our dataset
print(Y.value_counts())

# Optimizing or normalizing our data with standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit(X).transform(X.astype(float))
X[0:5]

# Spliting our data into training and testing part
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=4)
print(x_train.shape,x_test.shape)

# Using KNN(K Nearest neighbour) method for muticlass classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cheacking for which k values our model has highest accuracy in testing set(K=9 has highest accuracy score)
for k in range(1,10):
  neigh=KNeighborsClassifier(n_neighbors=k)
  neigh.fit(x_train,y_train)
  yhat=neigh.predict(x_test)
  print("accuracy for k=",k," is ",accuracy_score(yhat,y_test))

neigh=KNeighborsClassifier(n_neighbors=9)
neigh.fit(x_train,y_train)
yhat=neigh.predict(x_test)

# Display the traing and testing accuracy for our model 
print("accuracy for training is ",accuracy_score(y_train,neigh.predict(x_train)))
print("accuracy for testing is ",accuracy_score(yhat,y_test))






