import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data=load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
mle=LogisticRegression(C=np.inf, max_iter=1000)
mle.fit(X_train,y_train)
y_pred_mle=mle.predict(X_test)
print("MLE")
print("Accuracy:",accuracy_score(y_test,y_pred_mle))
print("Precision:",precision_score(y_test,y_pred_mle))
print("Recall:",recall_score(y_test,y_pred_mle))
print("F1 Score:",f1_score(y_test,y_pred_mle))
print("\n")

l1=LogisticRegression(l1_ratio=1, solver="liblinear", C=1.0, max_iter=5000)
l1.fit(X_train,y_train)
y_pred_l1=l1.predict(X_test)
print("MAP L1")
print("Accuracy:",accuracy_score(y_test,y_pred_l1))
print("Precision:",precision_score(y_test,y_pred_l1))
print("Recall:",recall_score(y_test,y_pred_l1))
print("F1 Score:",f1_score(y_test,y_pred_l1))
print("\n")

l2=LogisticRegression(l1_ratio=0, C=1.0, max_iter=5000)
l2.fit(X_train,y_train)
y_pred_l2=l2.predict(X_test)
print("MAP L2")
print("Accuracy:",accuracy_score(y_test,y_pred_l2))
print("Precision:",precision_score(y_test,y_pred_l2))
print("Recall:",recall_score(y_test,y_pred_l2))
print("F1 Score:",f1_score(y_test,y_pred_l2))
print("\n")
