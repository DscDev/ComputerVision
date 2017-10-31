# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#np.set_printoptions(threshold=np.inf)

# 1.Importing the dataset
dataset = pd.read_csv('data.csv')

## Todas las filas (:), Todas las columnas menos la utima (:-1)
x = dataset.iloc[:,:-1].values
## Todas las filas (:), La utima columna (-1)
y = dataset.iloc[:,-1].values

# 2.Taking care of missing data

from sklearn.preprocessing import Imputer

## missing_values, se ocupara de los valores que sean NaN (es 'NaN' por default)
## strategy, la estrategia que usara para rellenar esos campos  (es 'mean' por default)
## axis, si calculara en base a (0, las columnas) o (1, las filas) (es 0 por defecto) 
## imputer = Imputer(missing_values = 'NaN' , strategy = 'mean', axis = 0)

imputer = Imputer()
imputer = imputer.fit(x[:,1:])
x[:,1:] = imputer.transform(x[:,1:])

# 3.Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## categorical_features, que columnas queremos categorizar
onehotencoder = OneHotEncoder(categorical_features= [0])

x = onehotencoder.fit_transform(x).toarray();


# 4.Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

## random_state, ejemplos random solo para fines educativos
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

# 5.Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

