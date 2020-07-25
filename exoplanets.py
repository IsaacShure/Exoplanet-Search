"""
exoplanets.py

by Isaac Shure
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


import matplotlib.pyplot as plt
plt.close('all')
trainset = pd.read_csv('exoTrain.csv')
testset = pd.read_csv('exoTest.csv')

"""
t = range(0, 3198)
fig, ax = plt.subplots()
ax.plot(t, trainset.iloc[1], label="row 1")
ax.plot(t, trainset.iloc[2], label="row 2")
ax.plot(t, trainset.iloc[3], label="row 3")
ax.legend()

five = trainset[:5]
"""

# split dataframes into x and y components
y_train = trainset[['LABEL']]
x_train = trainset.drop(['LABEL'], axis=1)

y_test = testset[['LABEL']]
x_test = testset.drop(['LABEL'], axis=1)

# use logistic regression to predict exoplanets
# score: 0.50877 (model did not converge)

# use random forest to predict exoplanets
model = RandomForestClassifier()
model.fit(x_train, y_train.values.ravel())
predictions = model.predict(x_test)
score = model.score(x_test, y_test.values.ravel())

"""
model = Sequential()
model.add(Embedding())
"""
