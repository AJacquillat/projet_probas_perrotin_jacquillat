# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:03:58 2020

@author: Perrotin
"""

#Chargement de dépendances

import numpy as np
import matplotlib.pyplot as plt

#Discrétisation
A=0
B=500
N=101 #Nombre de point de discrétisation
Delta=(B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes * Delta

#Paramètres du modèle

mu = -5
a = 50
sigma2 = 12

#☻Données

observation_indexes = [0,20,40,60,80,100]
dpth = np.array([0 ,-4 ,-12.8 ,-1 ,-6.5 ,0])

#Indices des composantes correspondant aux observations et aux composantes non observées

unknown_indexes = list(set(discretization_indexes)-set(observation_indexes))

##1

def cov(dist,a,sigma2):
    distances=np.array(dist)
    n= np.shape(distances)[0] #la matrice de distance doit être carrée
    mat_cov=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mat_cov[i,j]=sigma2*np.exp(-np.abs(distances[i,j])/a)
    return mat_cov
    
##2

distances=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        distances[i,j] = Delta*np.abs(i-j)
print('Matrice des distances :', distances)
    
##3

'''
Il s'agit de la matrice donnée par la fonction cov avec pour entrée
la matrice distances
'''

cov_Z=cov(distances,a,sigma2)
print('Matrice de covariance de Z :', cov_Z)   

##4

'''
 - entre les observations : on retiendra  de la matrice précédente que les indices en ligne et en colonne contenu
 dans observation_indexes
'''
l=len(observation_indexes)

cov_observation=np.zeros((l,l))

for i in range(l):
    for j in range(l):
        cov_observation[i,j]=cov_Z[observation_indexes[i],observation_indexes[j]]

print('Matrice de covariance entre les observations :\n',cov_observation)

'''
 - entre les inconnues : idem
'''
l=len(unknown_indexes)

cov_inc=np.zeros((l,l))

for i in range(l):
    for j in range(l):
        cov_inc[i,j]=cov_Z[unknown_indexes[i],unknown_indexes[j]]

print('Matrice de covariance entre les inconnues :\n',cov_inc)
    
'''
 - entre les observations et les inconnues : on va mettre des zeros sur la matrice
cov_Z là où on a déjà pris des valeurs 
'''

l1=len(observation_indexes)
l2=len(unknown_indexes)
cov_inc_and_obs=np.zeros((l1+l2,l1+l2))

for i in range(l1):
    for j in range(l2):
        cov_inc_and_obs[i,j]=cov_Z[observation_indexes[i],unknown_indexes[j]]

for i in range(l2):
    for j in range(l1):
        cov_inc_and_obs[i,j]=cov_Z[unknown_indexes[i],observation_indexes[j]]


print('Matrice de covariance entre les observations et les inconnues :\n',cov_inc_and_obs)    


##5

    