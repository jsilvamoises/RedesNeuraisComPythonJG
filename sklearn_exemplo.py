# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:42:45 2019

@author: Usuario
"""
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

RELU = "relu"
IDENTITY= "identity"
LOGISTIC = "logistic"
TANH = "tanh"
iris = datasets.load_iris()

entradas = iris.data
saidas = iris.target

 redeNeural = MLPClassifier(activation=RELU,verbose=True, max_iter=10000,tol=0.00000000000001,learning_rate_init=0.01)
#redeNeural = MLPClassifier(verbose=True)
redeNeural.fit(entradas,saidas)

redeNeural.predict([[5,7.2,5.1,2.2]])




