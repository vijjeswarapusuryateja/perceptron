# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 02:29:10 2022

@author: Surya
"""

import pandas as pd

data = pd.read_excel('loan_details.xlsx')

df = data.copy()

train = df.iloc[:11, :]

test = df.iloc[10:, :]

X = df.drop('status', axis=1)

Y = df['status']

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)



import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

ann.fit(X, Y, epochs=5, verbose=1)

test_x = test.drop('status', axis=1)

test_x = ss.transform(test_x)

results = ann.predict(test_x)

