#14
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("mnist_784.csv")

X = data.drop("class",axis=1)
Y = data["class"]

X = X[:10000]
Y = Y[:10000]

X = X / 255.0

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

fns  = ["relu", "logistic", "tanh"]

for a in fns:
    model = MLPClassifier( hidden_layer_sizes = (50,), activation=a,  max_iter=1000)
    start = time.time()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    end = time.time()

    print("activation fn: ",a)
    print("accuracy: ",accuracy_score(y_test,pred))
    print("time: ",end - start)






