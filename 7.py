import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("20news_train.csv")
d = pd.read_csv("20news_test.csv")

data = data.dropna()
d = d.dropna()

X_train = data['text']
X_test = d['text']
y_train= data['label']
y_test = d['label']

vectorizer = CountVectorizer()
X_train_v = vectorizer.fit_transform(X_train)
X_test_v = vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_v, y_train)
y_pred_mnb = mnb.predict(X_test_v)

accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test,y_pred_mnb,average='weighted')

print("MULTINOMIAL NAIVE BAYES")
print("Accuracy:", accuracy_mnb)
print("F1-score:", f1_mnb)

binary_vectorizer = CountVectorizer(binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)

accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
f1_bnb = f1_score(y_test,y_pred_bnb,average='weighted')

print("\nBERNOULLI NAIVE BAYES")
print("Accuracy:", accuracy_bnb)
print("F1-score:", f1_bnb)
