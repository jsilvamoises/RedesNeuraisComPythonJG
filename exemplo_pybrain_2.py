# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:10:12 2019
pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip
@author: Usuario
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

## Não funcionou só deu erro
rede = buildNetwork(2, 3 ,1)
