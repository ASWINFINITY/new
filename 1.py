import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score


data=pd.read_csv('california_housing.csv')
X=data[['AveRooms']]
y=data['MedHouseVal']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


X_train=np.c_[np.ones(len(X_train)),X_train]
X_test=np.c_[np.ones(len(X_test)),X_test]
theta_gd=np.zeros(2)
rate=0.01
iterations=10000
m=len(y_train)


for i in range(iterations):
    y_pred= X_train @ theta_gd
    gradient= (2/m) * X_train.T @ (y_pred-y_train)
    theta_gd= theta_gd - rate * gradient
y_pred_gd= X_test @ theta_gd
print("GRADIENT DESCENT")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2 Score:", r2_score(y_test, y_pred_gd))


theta_ne=np.linalg.pinv(X_train.T@X_train)@ X_train.T @ y_train
y_pred_ne= X_test @ theta_ne
print("NORMAL EQUATION")
print("MSE:",mean_squared_error(y_test, y_pred_ne))
print("R2 SCORE:",r2_score(y_test, y_pred_ne))


plt.scatter(X_test[:,1],y_test, label="Actual")
plt.plot(X_test[:,1],y_pred_gd, color="red", label="Gradient Descent")
plt.plot(X_test[:,1],y_pred_ne, color="green", label="Normal Equation")
plt.legend()
plt.show()
