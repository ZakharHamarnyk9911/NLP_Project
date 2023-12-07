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
from nltk.corpus import stopwords

#load Data
filename = "Youtube03-LMFAO.csv"
path = "C:/Users\zhama\AI assignments\Group3_COMP237_ProjectNLP"
file_path = os.path.join(path, filename)

group_3 = pd.read_csv(file_path)


#data exploration
print(group_3.head(3))
print(group_3.columns)

#drop colums and prepare the data for model building
group_3_to_drop = ['COMMENT_ID', 'AUTHOR', 'DATE']
training_data = group_3.drop(group_3_to_drop, axis = 1)
print(training_data)

#shuffle the data
df_shuffled = training_data.sample(frac = 1, random_state = 1)

#prepare the data
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
train_data = count_vectorizer.fit_transform(df_shuffled['CONTENT'])

print(type(train_data))
print(train_data.shape)
names = count_vectorizer.get_feature_names_out()
print(names[:10])


#Downscale the transformed data using td-idf
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_data)

print(type(train_tfidf))
print(train_tfidf.shape)

#split date using pandas
split_data = int(0.75 * len(group_3))
X_train , Y_train = train_tfidf[:split_data], df_shuffled['CLASS'][:split_data]
X_test , Y_test =train_tfidf[:split_data], df_shuffled['CLASS'][:split_data]
print(X_train,Y_train)
print(X_test,Y_test)



#fit the classifier with the training data
classifier = MultinomialNB()
classifier.fit(X_train,Y_train)
#cross-validate the model
scores = cross_val_score(classifier,X_train , Y_train, cv=5)
#Test the modle on the test data
mean_accuracy = np.mean(scores)
print("Mean Accuracy:", mean_accuracy)


#make predictions on the test data
pred = classifier.predict(X_test)

#calculataion of the confusion matrix
calc_confusion_matrix = confusion_matrix(Y_test, pred)
print("Matirx:", calc_confusion_matrix)
#calculate accuracy
accuracy = accuracy_score(Y_test,pred)
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
    "Nice video, I love you so much guys",
    "Congratulations! You've been selected for a free gift card. Click now to claim!",
    "Boost your followers instantly! Click here for more details.",
    "This is fantastic! Keep up the amazing job!"
    
    ]
#transform vectorized data using tfidf transformer
input_tc = count_vectorizer.transform(test_data_comments)
input_tfidf = tfidf.transform(input_tc)

#predict the output
predictions = classifier.predict(input_tfidf)

# Print the outputs
for comment, predictions in zip(test_data_comments, predictions):
    print(f"\nComment Input:'{comment}' \nPredictions: {'Spam' if predictions == 1 else 'Not Spam'}")
