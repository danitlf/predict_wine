#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: danieltavaresdelimafreitas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('winemag-data-130k-v2.csv')

descriptions = df["description"].astype('str')
variaty =  df["variety"].astype('str')
country = df["country"].astype('str')

Y = country
train = descriptions + " " + variaty

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

clf_tree = tree.DecisionTreeClassifier()
clf_random = RandomForestClassifier()
#clf_svm = SVC(gamma='auto')
#gnb = GaussianNB()

print("start training")
clf_tree = clf_tree.fit(X_train, y_train)
clf_random = clf_random.fit(X_train, y_train)
#clf_svm = clf_svm.fit(X_train, y_train)
#gnb = gnb.fit(X_train.toarray(), y_train)


print("finish the training")
print("score Decision Tree")
print(clf_tree.score(X_test, y_test))

print("score Random Forest")
print(clf_random.score(X_test, y_test))

#print("score SVM")
#print(clf_svm.score(X_test, y_test))

#print("score Naive Bayes")
#print(gnb.score(X_test, y_test))