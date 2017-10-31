# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:10:10 2017

@author: dscxd
"""

from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network
import numpy as np
from matplotlib import pyplot as plt

_lr = linear_model.LinearRegression()
_svm = svm.SVC()
_nn = neural_network.MLPRegressor(max_iter=100)

arr_nums_train = range(1,10,2)
arr_nums_test = range(0,10,2)

x_train = np.transpose([arr_nums_train])
y_train = np.power(arr_nums_train,2)

x_test = np.transpose([arr_nums_test])

_lr.fit(x_train,y_train)
#_svm.fit(x_train,y_train)
#_nn.fit(x_train,y_train)

val_to_predict = 6

lr_prediction  = _lr.predict(x_test)
#svm_prediction = _svm.predict(x_test)
#nn_prediction  = _nn.predict(x_test)

#print("prediccion rl", lr_prediction)
#print("prediccion svm", svm_prediction)
#print("prediccion nn", nn_prediction)

plt.scatter(x_train, y_train)
plt.plot(x_test, lr_prediction, color='r')

import sys

print(sys.executable)
