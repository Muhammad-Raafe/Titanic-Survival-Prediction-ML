import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

df=pd.read_csv("Titanic-Dataset.csv")
print(df.info())


df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])


le=LabelEncoder()
df["Sex"]=le.fit_transform(df["Sex"])
df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)



x=df.drop(["PassengerId","Name","Cabin","Ticket","Survived"],axis=1)
y=df["Survived"]


x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


model=KNeighborsClassifier()


param_grid={
    "n_neighbors":[3,5,7,9,13,15],
    "weights":["uniform","distance"],
    "metric":["euclidean","manhattan"]
}





grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(x_train,y_train)
prediction=grid.predict(x_test)

print("Cross Validation Score Is",grid.best_score_)
print("Best Parameters Are: ",grid.best_params_)
print("Accuracy Score Is: ",accuracy_score(y_test,prediction))
print("Confusion Matrix: ",confusion_matrix(y_test,prediction))
print("Classification Report",classification_report(y_test,prediction))

