# LINEAR SVM
# Iris Dataset
# Setosa vs Non-Setosa

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. LOAD DATASET
iris = load_iris()
X = iris.data 
y = iris.target

# 2. CONVERT INTO BINARY CLASSIFICATION
# Setosa = 1
# Non-Setosa = 0
y = (y == 0).astype(int)

# 3. SELECT TWO FEATURES
# Feature 0 = Sepal Length
# Feature 1 = Sepal Width
X = X[:, [0, 1]]

# 4. CREATE LINEAR SVM
svm = SVC(kernel="linear", C=1)
svm.fit(X, y)

# 5. PREDICT
y_pred = svm.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# 6. GET DECISION BOUNDARY
w = svm.coef_[0]
b = svm.intercept_[0]

x_values = np.array([X[:, 0].min(), X[:, 0].max()])

# Decision boundary: w0*x + w1*y + b = 0
y_values = -(w[0] * x_values + b) / w[1]

# 7. GET MARGIN LINES
# Upper margin: w0*x + w1*y + b = 1
y_upper = -(w[0] * x_values + b - 1) / w[1]

# Lower margin: w0*x + w1*y + b = -1
y_lower = -(w[0] * x_values + b + 1) / w[1]

# 8. VISUALIZE
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Setosa")
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Non-Setosa")

plt.plot(x_values, y_values, label="Decision Boundary")
plt.plot(x_values, y_upper, "--", label="Margin")
plt.plot(x_values, y_lower, "--")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()
