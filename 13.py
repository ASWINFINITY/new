import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv("winequality-red.csv")
X=data.drop("quality", axis=1)
y=data["quality"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model1=MLPClassifier(hidden_layer_sizes=(50,),max_iter=5000, random_state=42)
model1.fit(X_train_scaled,y_train)
y_pred1=model1.predict(X_test_scaled)
print("Accuary m1:", accuracy_score(y_test,y_pred1))
model2=MLPClassifier(hidden_layer_sizes=(50,25),max_iter=5000, random_state=42)
model2.fit(X_train_scaled,y_train)
y_pred2=model2.predict(X_test_scaled)
print("Accuary m2:", accuracy_score(y_test,y_pred2))

