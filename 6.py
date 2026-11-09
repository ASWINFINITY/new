# Program 6: Logistic Regression for Diabetes Prediction

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset

df = pd.read_csv('pima_indians_diabetes.csv')

# Step 2: Separate input features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train Logistic Regression WITHOUT feature scaling
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

# Evaluate Model 1
print("WITHOUT FEATURE SCALING")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(y_test, y_pred1, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred1, zero_division=0))

# Step 5: Scale the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression WITH feature scaling
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)

# Evaluate Model 2
print("\nWITH FEATURE SCALING")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(y_test, y_pred2, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred2, zero_division=0))
