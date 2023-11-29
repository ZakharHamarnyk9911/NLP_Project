# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:05:27 2023

@author: Zakhar Hamarnyk -301250102
"""

import pandas as pd
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#load Data
filename = "Youtube03-LMFAO.csv"
path = "C:/Users\zhama\AI assignments\Group3_COMP237_ProjectNLP"
file_path = os.path.join(path, filename)

group_3 = pd.read_csv(file_path)

#drop colums
group_3 = group_3.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)
print(group_3)

#split date using pandas
split_data = int(0.75 * len(group_3))
train_data = group_3[:split_data]
test_data = group_3[split_data:]
print(train_data)
print(test_data)

#seperate the features
X_train = train_data['CONTENT']
y_train = train_data['CLASS']
X_test = test_data['CONTENT']
y_test = test_data['CLASS']
#vectorize the training data
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_tc)

#vectorize the test data transform only
test_tc = count_vectorizer.transform(X_test)
tfidf = tfidf_transformer.transform(test_tc)

#cross-validate the model
classifier = MultinomialNB()
scores = cross_val_score(classifier, train_tfidf, y_train, cv=5)
print("Accuracy of each fold:", scores)
mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)

#fit the classifier with the training data
classifier.fit(train_tfidf, y_train)

#make predictions on the test data
y_pred = classifier.predict(tfidf)

#calculataion of the confusion matrix
calc_confusion_matrix = confusion_matrix(y_test, y_pred)
print(calc_confusion_matrix)
#calculate accuracy
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy is :", accuracy)

#test our comments
test_data_comments =[
    "This video is nice, I love it , I need more of them",
    "Click here to win a new Iphone 15Pro Max",
    "Bill Gates gives everyone money for free",
    "Guys you inspire and motivate me , keep it up!",
    "Elon Musk share 0.50 BTC, folow the link and het your money for free",
    "Good job, appreceit it",
    "Really insightful analysis, thanks for sharing!",
    "Your explanation made this topic so clear, great job!",
    "Hot singles in your area waiting to meet you! Click now!"
    ]
input_tc = count_vectorizer.transform(test_data_comments)
type(input_tc)
print(input_tc)
#transform vectorized data using tfidf transformer
input_tfidf = tfidf_transformer.transform(input_tc)
type(input_tfidf)
print(input_tfidf)
#predict the output
predictions = classifier.predict(input_tfidf)

# Print the outputs
for comment, predictions in zip(test_data_comments, predictions):
    print(f"\nComment Input:'{comment}' \nPredictions: {'Spam' if predictions == 1 else 'Not Spam'}")
