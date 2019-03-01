# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:25:44 2019
time.sleep(1)
@author: Moisés
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
class RedeMultiLayer():
    def __init__(self,entradas,saidas,epocas = 10000, taxaAprendizado=0.3):
        self.entradas = entradas
        self.saidas = saidas
        #self.pesosEntrada = np.random.random((2,3)) #np.array([[0.424,-0.740,-0.967],[0.358,-0.577,-0.469]])
        #self.pesosSaida = np.random.random((3,1)) #np.array([[-0.017],[-0.893],[0.148]])
        
        self.pesosEntrada = np.random.random((entradas.shape[1],entradas.shape[0]-1)) #np.array([[0.424,-0.740,-0.967],[0.358,-0.577,-0.469]])
        self.pesosSaida = np.random.random((saidas.shape[0]-1,saidas.shape[1])) #np.array([[-0.017],[-0.893],[0.148]])
        
        
        self.epocas = epocas
        self.taxaAprendizado = taxaAprendizado
        self.momento = 1
        self.erros = []
        
    # ==========================================================
    """
    x = soma
    y = _____1____
          1 + exp-x          
    """
    def sigmoid(self,soma):
        return 1 / (1 + np.exp(-soma))
    # ==========================================================
    def sigmoidDerivada(self,sig):
        return sig * (1 - sig)
    # ==========================================================
    def initTreinamento(self):
        for i in range(self.epocas):
            #msg = "Epoca.: ",i
            #print(msg,end="")
            #print("\b"*(len(msg)),end="\r",flush = True)
            camadaEntrada = self.entradas
            somaSinapseEntrada = np.dot(camadaEntrada,self.pesosEntrada)
            camadaOculta = self.sigmoid(somaSinapseEntrada)
            
            somaSinapseEntrada = np.dot(camadaOculta,self.pesosSaida)
            camadaSaida = self.sigmoid(somaSinapseEntrada)
            
            erroCamadaSaida = self.saidas - camadaSaida
            mediaAbs = np.mean(np.abs(erroCamadaSaida))
            self.erros.append(mediaAbs)
            print("Erros.: ",mediaAbs)
            
            
            derivadaSaida = self.sigmoidDerivada(camadaSaida)
            deltaSaida = erroCamadaSaida * derivadaSaida
            
            pesosATranspost = self.pesosSaida.T # Transposta
            deltaSaidaXPeso = deltaSaida.dot(pesosATranspost)            
            deltaCamadaOculta = deltaSaidaXPeso * self.sigmoidDerivada(camadaOculta)
            
            
            # atualizada pesos camada oculta para camada saida
            camadaOcultaTransposta = camadaOculta.T # Transposta
            pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)            
            self.pesosSaida = (self.pesosSaida * self.momento)+(pesosNovo1 * self.taxaAprendizado)
            
            # Atualiza pesos da camada de entrada para oculta
            camadaEntradaTransposta = camadaEntrada.T
            pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
            self.pesosEntrada = (self.pesosEntrada * self.momento)+(pesosNovo0 * self.taxaAprendizado)
            
            #print(self.pesosSaida,"\n")
        
    def plotChart(self):
        plt.plot(self.erros)
        plt.show()
  



"""

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]]) 
rm = RedeMultiLayer(entradas,saidas)
rm.initTreinamento()
# 2,3
print(entradas.shape[1]) 
print(entradas.shape[0]-1)  
 

#3,1      

print(saidas.shape[0]-1 )  
print(saidas.shape[1])          

rm = RedeMultiLayer(entradas,saidas)
rm.initTreinamento()

plt.plot(rm.erros)
"""





