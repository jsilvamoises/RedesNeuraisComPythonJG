# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:15:44 2019

@author: Usuario
"""

import pandas as pd
import numpy as np
from rede_neural_multicamada import RedeMultiLayer
from sklearn import datasets
dataframe = datasets.load_breast_cancer()

entradas = dataframe.data
valoresSaida = dataframe.target
saidas = np.empty([569,1],dtype=int)

for i in range(len(valoresSaida)):
    saidas[i] = valoresSaida[i]

print(entradas.shape)
print(saidas.shape)

rn = RedeMultiLayer(entradas,saidas,epocas=10000,taxaAprendizado=0.9)
rn.initTreinamento()
rn.plotChart()


entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]]) 
rm = RedeMultiLayer(entradas,saidas,epocas=10000)
rm.initTreinamento()
rn.plotChart()