import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time


train_data=pd.read_csv("fashion-mnist_train.csv")
test_data=pd.read_csv("fashion-mnist_test.csv")


X_train=train_data.drop("label", axis=1)[:10000]
X_test=test_data.drop("label", axis=1)[:10000]
y_train=train_data['label'][:10000]
y_test=test_data['label'][:10000]
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
kernels=['linear','poly','rbf']
for kernel in kernels:
    print(kernel)
    start=time.time()
    model=SVC(kernel=kernel,C=1)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    end=time.time()
    accuracy= accuracy_score(y_test,y_pred)
    print("Accuracy:", accuracy)
    print("Time:", end - start)

