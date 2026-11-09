from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time


train_data=pd.read_csv("fashion-mnist_train.csv")
test_data=pd.read_csv("fashion-mnist_test.csv")


X_train=train_data.drop("label", axis=1)[:10000]
X_test=test_data.drop("label", axis=1)[:10000]
y_train=train_data['label'][:10000]
y_test=test_data['label'][:10000]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
settings=[(0.001,32,80),(0.01,32,80),(1.001,64,80)]
for learning_rate,batch_size,epochs in settings:
    print("\nLearning Rate:", learning_rate)
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    start=time.time()
    model=MLPClassifier(hidden_layer_sizes=(50,),activation='relu',learning_rate_init=learning_rate,batch_size=batch_size,max_iter=epochs,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    end=time.time()
    accuracy= accuracy_score(y_test,y_pred)
    print("Accuracy",accuracy)
    print("Time",end-start)


