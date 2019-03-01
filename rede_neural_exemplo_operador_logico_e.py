# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:33:18 2019

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt

class RedeNeural():
    def __init__(self,entradas,saidas,taxaAprendizagem=0.1):
        self.entradas = entradas
        self.saidas = saidas
        self.pesos = np.array([0.0,0.0])
        self.taxaAprendizagem = taxaAprendizagem
        self.arrayPesos= []
        
    def stepFunction(self,soma):
        if (soma >= 1):
            return 1
        return 0
    
    def calculaSaida(self,registro):
        s = registro.dot(self.pesos)
        r = self.stepFunction(s)
        return r
    
    def treinar(self):
        
        erroTotal = 1
        while(erroTotal!=0):
            erroTotal = 0
            for i in range(len(self.saidas)):
                saidaCalculada = self.calculaSaida(np.array(entradas[i]))
                erro = abs(self.saidas[i] - saidaCalculada)
                erroTotal+=erro
                
                for j in range(len(self.pesos)):
                    self.pesos[j] = self.pesos[j]+(self.taxaAprendizagem * self.entradas[i][j] * erro)
                    self.arrayPesos.append(self.pesos[j])
                    plt.plot(self.arrayPesos)
                    
                    
                    print("Peso Atualizado",str(self.pesos[j]))
                   
            print("Total de Erros",str(erroTotal))
        plt.title("Pesos")
        plt.show() 
        
        
        
        
        





## REDE NEURAL TEXTE E
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,0,0,1])
rn = RedeNeural(entradas,saidas)
rn.treinar()

print(rn.calculaSaida(entradas[0]))
print(rn.calculaSaida(entradas[1]))
print(rn.calculaSaida(entradas[2]))
print(rn.calculaSaida(entradas[3]))

## REDE NEURAL TEXTE OR
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])
rn = RedeNeural(entradas,saidas)
rn.treinar()

print(rn.calculaSaida(entradas[0]))
print(rn.calculaSaida(entradas[1]))
print(rn.calculaSaida(entradas[2]))
print(rn.calculaSaida(entradas[3]))