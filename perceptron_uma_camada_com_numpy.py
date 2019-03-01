# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:23:41 2019

@author: Usuario
"""
import numpy as np

class PerceptronUmaCamada():
    def __init__(self,entradas,pesos):
        self.entradas = entradas
        self.pesos = pesos
        
    def soma(self,e,p):
        return e.dot(p)
    
    def stepFunction(self,soma):
        if (soma >= 1):
            return 1
        return 0
        
























entradas = np.array([1,7,5])
pesos = np.array([0.8,0.1,0])

pc = PerceptronUmaCamada(entradas,pesos)
s = pc.soma(entradas,pesos)
r = pc.stepFunction(s)