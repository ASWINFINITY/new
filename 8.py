import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


train_data=pd.read_csv("fashion-mnist_train.csv")
test_data=pd.read_csv("fashion-mnist_test.csv")


X_train=train_data.drop("label", axis=1)[:10000]
X_test=test_data.drop("label", axis=1)[:10000]
y_train=train_data['label'][:10000]
y_test=test_data['label'][:10000]


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


values=[3,5,7]


for k in values:
    start=time.time()
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    end=time.time()
    accuracy=accuracy_score(y_test,y_pred)
    print("K=",k)
    print("Accuracy",accuracy)
    print("TIme:",end-start)
    print()
