# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:23:37 2019

@author: Moises Silva
"""



class PerceptronUmaCamada():
    def __init__(self,entradas,pesos):
        self.entradas = entradas
        self.pesos = pesos
        
    def soma(self,e,p):
        s = 0
        for i in range(len(self.entradas)):
            #print(self.entradas[i])
            #print(self.pesos[i])
            s += e[i] * p[i]
            
        return s
    
    def stepFunction(self,soma):
        if (soma >= 1):
            return 1
        else:
            return 0
        


entradas = [-1,7,5]
pesos = [0.8,0.1,0]

pc = PerceptronUmaCamada(entradas,pesos)
s = pc.soma(entradas,pesos)
r = pc.stepFunction(s)