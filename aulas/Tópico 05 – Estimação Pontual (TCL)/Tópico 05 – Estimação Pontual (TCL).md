---
layout: default
title: "Tópico 05 – Estimação Pontual (TCL)"
parent: "Aulas"
nav_order: 5
---
# Tópico 05 – Estimação Pontual (TCL) [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29/images/colag_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 6.5%;">](https://colab.research.google.com/github/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29.ipynb) [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29/images/github_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 3.25%;">](https://github.com/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29.ipynb)




### MATERIAL EM CONSTRUÇÃO [!]

Material adaptado do [DSC10 (UCSD)](https://dsc10.com/) por [Flavio Figueiredo (DCC-UFMG)](https://flaviovdf.io/fcd/) e [Uriel Silva (DEST-UFMG)](https://urielmoreirasilva.github.io)


```python
## Imports para esse tópico
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
import math as math
plt.style.use('ggplot')

## Opções de como printar objetos do Numpy e do Pandas
np.set_printoptions(threshold = 20, precision = 2, suppress = True)
pd.set_option("display.max_rows", 7)
pd.set_option("display.max_columns", 8)
pd.set_option("display.precision", 2)
```

## Exemplo #1: $Poisson(\lambda)$


```python
"""
Exemplo #1: Y_i ~ iid. Poisson(lambda)
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho amostral
n = int(1e3)

## Quantidade de amostras independentes
m = int(1e3)

## Parâmetros da simulação
lam = 2 # taxa da Poisson
mu = 2 # média populacional

## Declarando objetos
vY_bar = np.zeros(m)

## Loop principal
for j in np.arange(0, m, 1):
    vY = np.random.poisson(lam = lam, size = n) # gerando amostra
    vY_bar[j] = vY.mean() # calculando a média
```


```python
## Amostra das médias
vY_bar
```




    array([2.  , 1.98, 1.9 , ..., 1.98, 2.03, 2.02], shape=(1000,))




```python
## Visualizando a distribuição amostral das médias
plt.figure(figsize = (12, 6))
plt.hist(vY_bar, 
         bins = np.arange(vY_bar.min(), vY_bar.max(), (vY_bar.max() - vY_bar.min())/20), 
         density = True, 
         ec = 'w')
plt.axvline(mu, color = 'black', linewidth = 4, label = f'Média populacional ($\mu$ = {lam:.2f})')
plt.title('Teorema Central do Limite: Poisson', fontsize = 14)
plt.xlabel('Médias', fontsize = 12)
plt.ylabel('Densidade', fontsize = 12)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29_files/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29_6_0.png)
    


## Exemplo #1: $Beta(\alpha, 1)$


```python
"""
Exemplo #1: Y_i ~ iid. Beta(\alpha, 1)
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho amostral
n = int(1e3)

## Parâmetros da simulação
alpha = 1./2 # forma da Beta
beta = 1 # outro parâmetro de forma da Beta :)
mu = alpha / (alpha + beta) # média populacional

## Declarando objetos
vY_bar = np.zeros(m)

## Loop principal
for j in np.arange(0, m, 1):
    vY = np.random.beta(a = alpha, b = 1, size = n) # gerando amostra
    vY_bar[j] = vY.mean() # calculando a média
```


```python
## Amostra das médias
vY_bar
```




    array([0.32, 0.34, 0.34, ..., 0.35, 0.33, 0.35], shape=(1000,))




```python
## Visualizando a distribuição amostral das médias
plt.figure(figsize = (12, 6))
plt.hist(vY_bar, 
         bins = np.arange(vY_bar.min(), vY_bar.max(), (vY_bar.max() - vY_bar.min())/20), 
         density = True, 
         ec = 'w')
plt.axvline(mu, color = 'black', linewidth = 4, label = f'Média populacional ($\mu$ = {mu:.2f})')
plt.title('Teorema Central do Limite: Beta($\\alpha$, 1)', fontsize = 14)
plt.xlabel('EMVs', fontsize = 12)
plt.ylabel('Densidade', fontsize = 12)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29_files/T%C3%B3pico%2005%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28TCL%29_10_0.png)
    

