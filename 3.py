import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv('diabetes.csv')
X=data.drop('target', axis=1)
y=data['target']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


linear=LinearRegression()
linear.fit(X_train,y_train)
y_pred_lin=linear.predict(X_test)
print("LINEAR REGRESSION")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R2 Score:", r2_score(y_test, y_pred_lin))


ridge=RidgeCV(alphas=[0.01,0.1,1,10,100],cv=5)
ridge.fit(X_train,y_train)
y_pred_r=ridge.predict(X_test)
print("\nRIDGE REGRESSION")
print("BEST ALPHA:",ridge.alpha_)
print("MSE:",mean_squared_error(y_test,y_pred_r))
print("R2 SCORE:", r2_score(y_test,y_pred_r))


lasso=LassoCV(alphas=[0.01,0.1,1,10,100],cv=5,max_iter=10000)
lasso.fit(X_train,y_train)
y_pred_l=lasso.predict(X_test)
print("\nLASSO REGRESSION")
print("BEST ALPHA:", ridge.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred_l))
print("R2 Score:", r2_score(y_test, y_pred_l))


