import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv('auto-mpg.csv')
X= data[['displacement']]
y=data['mpg']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
linear=LinearRegression()
linear.fit(X_train,y_train)
y_pred_lin=linear.predict(X_test)
print("LINEAR REGRESSION")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R2 Score:", r2_score(y_test, y_pred_lin))
plt.scatter(X_test,y_test, label="Actual")
plt.plot(X_test,y_pred_lin, label="Linear Regression")
#argsort() → gives sorting indices.
idx=X_test["displacement"].argsort()
degrees=[2,3,4]
for d in degrees:
    poly=PolynomialFeatures(degree=d)
    X_train_p=poly.fit_transform(X_train)
    X_test_p=poly.transform(X_test)
    model=LinearRegression()
    model.fit(X_train_p,y_train)
    y_pred_poly=model.predict(X_test_p)
    #.iloc[idx] → arranges DataFrame/Series rows according to those indices.
    #y_pred_poly[idx] → arranges predictions in exactly the same order.
    plt.plot(X_test["displacement"].iloc[idx],y_pred_poly[idx], label="Polynomial Degree: "+ str(d))
    print("POLYNOMIAL REGRESSION DEGREE ",d)
    print("MSE:",mean_squared_error(y_test,y_pred_poly))
    print("R2 SCORE:",r2_score(y_test,y_pred_poly))
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Linear vs Polynomial")
plt.legend()
plt.show()

