# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:30:53 2019
Base de dados para testes: https://archive.ics.uci.edu/ml/index.php
@author: Usuario
"""
import pandas as pd
import numpy as np
from rede_neural_multicamada import RedeMultiLayer

dados = pd.read_csv("lotofacil.csv",sep=";")
dados = np.array(dados)
entradas = dados[:1200,:25]
saidas = dados[:1200,25:]

print(entradas.shape)
print(saidas.shape)

rn = RedeMultiLayer(entradas,saidas,epocas=200)
rn.initTreinamento()
rn.plotChart()