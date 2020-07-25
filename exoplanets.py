"""
exoplanets.py

by Isaac Shure
"""
import pandas as pd
import numpy as np

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
trainy = trainset[['LABEL']]
trainx = trainset.drop(['LABEL'], axis=1)

testy = testset[['LABEL']]
testx = trainset.drop(['LABEL'], axis=1)
#model = Sequential()
#model.add(Embedding())