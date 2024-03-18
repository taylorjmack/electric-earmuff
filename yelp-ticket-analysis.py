

# -*- coding: utf-8 -*-
"""
Author:Taylor Mack

The Yelp dataset that utilized in this analysis can be found https://yelp.com/dataset

The downloaded dataset contains 5 different json files business.json, review.json, user.json, checkin.json,
tip.json, and photo.json. This analysis was conducted on the review.json file in that download.

There are 2 models that are utilized to compare results

Model 1 is a RNN model done through keras, utilizing one-hot list to convert words into values

Model 2 is a logistical regression due, since the results are inherently binary due to the fact
that a review should be considered as either positive or negative

"""

import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

file = 'insert local path to your review.json file'

data = []
d_chunk = pd.read_json(file,lines=True,chunksize= 10000)
data = pd.concat([c for c in d_chunk])
print('Successfully Chunked the json file')

#data set has 6,990,280 rows parsing down data to just be 2021 reviews. You can expand the scope of the reviews if needed

data = data[(data['date'] >= '2021-01-01') & (data['date'] <= '2021-12-31')]

#data set is now at 616,220 rows after filter
#dropping 3-star reviews because they are neutral reviews
data = data[data['stars'] != 3]
data.reset_index(drop=True,inplace=True)
data.loc[data['stars'] > 3, 'Result'] = 1
data.loc[data['stars'] < 3, 'Result'] = 0
print('Successfully created stars binary classifer')
data = data.loc[(data['Result'] == 1) | (data['Result'] == 0)]

#You can filter down rows here if there are computing contraints for the model
row_count = 100000
df = data[:row_count]

text_list = list(df['text'])
one_hot_list = []

#Generating one hot list with 10,000 of the most popular words used
for t in text_list:
    one_hot_list.append(one_hot(t,10000))
print('Created one_hot_list')

#Parsed down reviews to be a max of 120 words long and coverting list with sublists to a numpy array
X = pad_sequences(one_hot_list,maxlen=120)
print('limited one_hot_list to 120 words')

#Converting the Results column on df DataFrame to a numpy array
Y = np.asarray(df['Result'])

n = len(df)
#Creating 80% train numpy array
x_train = X[:round(n*.8)]
y_train = Y[:round(n*.8)]
print('Created train dataset')

#Creating 20% test numpy array
x_test = X[-round(n*.2):]
y_test = Y[-round(n*.2):]
print('Created test dataset')

#Starting model generation process
print('Starting model')
mdl = Sequential()
print('Loaded Squential')

#10,000 of the most popular words and 128 nuerons
print('Starting model with the 10000 most popular words with 128 neurons')
mdl.add(Embedding(10000,128,input_length=120))
mdl.add(LSTM(128))
print('Successfully ran LSTM model')

mdl.add(Flatten())
print('Sucessfully ran Flatten model')

mdl.add(Dense(1, activation= 'sigmoid'))
print('Starting compile function with a binary crossentropy')

mdl.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','FalsePositives','FalseNegatives','TruePositives','TrueNegatives'])
mdl.fit(x_train, y_train)
print('Successfully ran Fitting Model')

print('Starting predict function')
pred_y = mdl.predict(x_test)
print('Ran predict function')

m_eval = mdl.evaluate(x_test,y_test)
#manually generating model 1 confusion matrix
c_matrix_1 = np.array([m_eval[4],m_eval[2]])
c_matrix_2 = np.array([m_eval[3],m_eval[5]])
c_matrix = np.vstack((c_matrix_1,c_matrix_2))
print(c_matrix)

#generating model 1 accuracy
accuracy = mdl.evaluate(x_test,y_test)[1]
print('Accuracy from One_Hot Model:',accuracy)

#due to processing issues parsing down data to 20,000 lines, this can be toned up or down depending on resourcing capabilities
df_s = data[:20000]

#starting model 2 process
c_vector = CountVectorizer(analyzer='word',lowercase=False)
x_vect = c_vector.fit_transform(df_s['text']).toarray()
y_sk = np.asarray(df_s['Result'])

#generating train and test datasets
sk_x_train,sk_x_test,sk_y_train,sk_y_test = train_test_split(x_vect,y_sk,train_size=.8)

#generating logistical regression
clf = LogisticRegression(max_iter=1000)
clf.fit(sk_x_train,sk_y_train)
sk_y_pred = clf.predict(sk_x_test)

#generating sklearn confusion matrix
sk_c_matrix = confusion_matrix(sk_y_test,sk_y_pred)
print(sk_c_matrix)

#generating model 2 accuracy
sk_accuracy = accuracy_score(sk_y_test,sk_y_pred)
print('Sklearn Accuracy:',sk_accuracy)