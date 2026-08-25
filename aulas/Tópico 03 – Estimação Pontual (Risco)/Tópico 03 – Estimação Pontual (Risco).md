---
layout: default
title: "Tópico 03 – Estimação Pontual (Risco)"
parent: "Aulas"
nav_order: 3
---
# Tópico 03 – Estimação Pontual (Risco) [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29/images/colag_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 6.5%;">](https://colab.research.google.com/github/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29.ipynb) [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29/images/github_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 3.25%;">](https://github.com/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29.ipynb)




### MATERIAL EM CONSTRUÇÃO [!]

Material adaptado do [DSC10 (UCSD)](https://dsc10.com/) por [Flavio Figueiredo (DCC-UFMG)](https://flaviovdf.io/fcd/) e [Uriel Silva (DEST-UFMG)](https://urielmoreirasilva.github.io)


```python
## Imports para esse tópico
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

## Opções de como printar objetos do Numpy e do Pandas
np.set_printoptions(threshold = 20, precision = 2, suppress = True)
pd.set_option("display.max_rows", 7)
pd.set_option("display.max_columns", 8)
pd.set_option("display.precision", 2)
```

## Exemplo #1: $N(0, 1)$


```python
"""
Exemplo #1: Y_i ~ iid. Normal(0, 1)
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho da sequência
n = int(1e3)

## Quantidade de sequências independentes
m = int(1e3)

## Parâmetros da simulação
mu = 0 # média populacional
sigma2 = 1 # variância populacional

## Declarando objetos
vY = np.zeros((n, m)) # v.a.'s
vY_bar = np.zeros((n, m)) # médias

## Loop principal
for j in np.arange(0, m, 1):
    for i in np.arange(0, n, 1):
        vY[i, j] = np.random.randn() 
        vY_bar[i, j] = vY[0:i+1, j].mean()
```


```python
## Vizinhança para o gráfico de convergência (experimente mudar esse valor!)
eps = 0.30

## Gráfico da convergência
plt.figure(figsize = (12, 6))
plt.plot(vY_bar)
plt.axhline(mu - eps, color = 'red', linestyle = '--')
plt.axhline(mu, color = 'black', linestyle = '--', label = f'Média populacional (μ = {mu:.2f})')
plt.axhline(mu + eps, color = 'red', linestyle = '--')
plt.title('Demonstração da WLLN', fontsize = 14)
plt.xlabel('Tamanho amostral (n)', fontsize = 12)
plt.ylabel('Sequências de Médias (Amostrais)', fontsize = 12)
plt.ylim(-2.0, 2.0)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_5_0.png)
    



```python
## Calculando EQMs e EAMs
vMSE = np.zeros(n) # EQMs
vMAE = np.zeros(n) # EAMs

## Loop principal
for i in np.arange(0, n, 1):
    vMSE[i] = ((vY_bar[i, :] - mu)**2).mean()
    vMAE[i] = (abs(vY_bar[i, :] - mu)).mean()
```


```python
## Gráfico da convergência do EQM e EAM
plt.figure(figsize = (12, 6))
plt.plot(vMSE, color = 'blue', label = 'MSE')
plt.plot(vMAE, color = 'red', label = 'MAE')
plt.axhline(0.00, color = 'black', linestyle = '--')
plt.title('Risco como função de n', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Estimativas dos riscos', fontsize = 12)
plt.ylim(-0.01, 0.10)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_7_0.png)
    



```python
## Checando o resultado analítico para o MSE
i = 10 # experimente mudar esse valor!
print(vMSE[i])
print(1 / (i + 1))
```

    0.08429197723417521
    0.09090909090909091



```python
## Checando o resultado analítico para o MAE
i = 10 # experimente mudar esse valor!
print(vMAE[i])
print(0.7978 * np.sqrt(1 / (i + 1)))
```

    0.22903206775108728
    0.2405457507041398


## Exemplo #2: $t_4$


```python
"""
Exemplo #2: Y_i ~ iid. t_4
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho da sequência
n = int(1e3)

## Quantidade de sequências independentes
m = int(20)

## Parâmetros da simulação
mu = 0 # média populacional
sigma2 = 1 # variância populacional
nu = 4 # graus de liberdade

## Declarando objetos
vY = np.zeros((n, m)) # v.a.'s
vY_bar = np.zeros((n, m)) # médias

## Loop principal
for j in np.arange(0, m, 1):
    for i in np.arange(0, n, 1):
        vY[i, j] = np.random.standard_t(df = nu) 
        vY_bar[i, j] = vY[0:i+1, j].mean()
```


```python
## Vizinhança para o gráfico de convergência (experimente mudar esse valor!)
eps = 0.30

## Gráfico da convergência
plt.figure(figsize = (12, 6))
plt.plot(vY_bar)
plt.axhline(mu - eps, color = 'red', linestyle = '--')
plt.axhline(mu, color = 'black', linestyle = '--', label = f'Média populacional (μ = {mu:.2f})')
plt.axhline(mu + eps, color = 'red', linestyle = '--')
plt.title('Demonstração da WLLN', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Sequências de Médias (Amostrais)', fontsize = 12)
plt.ylim(-2.0, 2.0)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_12_0.png)
    



```python
## Calculando EQMs e EAMs
vMSE = np.zeros(n) # EQMs
vMAE = np.zeros(n) # EAMs

## Loop principal
for i in np.arange(0, n, 1):
    vMSE[i] = ((vY_bar[i, :] - mu)**2).mean()
    vMAE[i] = (abs(vY_bar[i, :] - mu)).mean()
```


```python
## Gráfico da convergência do EQM e EAM
plt.figure(figsize = (12, 6))
plt.plot(vMSE, color = 'blue', label = 'MSE')
plt.plot(vMAE, color = 'red', label = 'MAE')
plt.axhline(0.00, color = 'black', linestyle = '--')
plt.title('Risco como função de n', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Estimativas dos riscos', fontsize = 12)
plt.ylim(-0.01, 0.10)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_14_0.png)
    


## Exemplo #3: $t_2$


```python
"""
Exemplo #3: Y_i ~ iid. t_2

NOTA: uma t-Student com 2 graus de liberdade tem variância infinita!
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho da sequência
n = int(1e3)

## Quantidade de sequências independentes
m = int(20)

## Parâmetros da simulação
mu = 0 # média populacional
sigma2 = 1 # variância populacional
nu = 2 # graus de liberdade

## Declarando objetos
vY = np.zeros((n, m)) # v.a.'s
vY_bar = np.zeros((n, m)) # médias

## Loop principal
for j in np.arange(0, m, 1):
    for i in np.arange(0, n, 1):
        vY[i, j] = np.random.standard_t(df = nu) 
        vY_bar[i, j] = vY[0:i+1, j].mean()
```


```python
## Vizinhança para o gráfico de convergência (experimente mudar esse valor!)
eps = 0.30

## Gráfico da convergência
plt.figure(figsize = (12, 6))
plt.plot(vY_bar)
plt.axhline(mu - eps, color = 'red', linestyle = '--')
plt.axhline(mu, color = 'black', linestyle = '--', label = f'Média populacional (μ = {mu:.2f})')
plt.axhline(mu + eps, color = 'red', linestyle = '--')
plt.title('Demonstração da WLLN', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Sequências de Médias (Amostrais)', fontsize = 12)
plt.ylim(-2.0, 2.0)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_17_0.png)
    



```python
## Calculando EQMs e EAMs
vMSE = np.zeros(n) # EQMs
vMAE = np.zeros(n) # EAMs

## Loop principal
for i in np.arange(0, n, 1):
    vMSE[i] = ((vY_bar[i, :] - mu)**2).mean()
    vMAE[i] = (abs(vY_bar[i, :] - mu)).mean()
```


```python
## Gráfico da convergência do EQM e EAM
plt.figure(figsize = (12, 6))
plt.plot(vMSE, color = 'blue', label = 'MSE')
plt.plot(vMAE, color = 'red', label = 'MAE')
plt.axhline(0.00, color = 'black', linestyle = '--')
plt.title('Risco como função de n', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Estimativas dos riscos', fontsize = 12)
plt.ylim(-0.01, 1)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_19_0.png)
    


## Exemplo #4: $t_1$


```python
"""
Exemplo #4: Y_i ~ iid. t_1

NOTA: a t-Student com 1 grau de liberdade é igual à Cauchy!
"""

## Semente aleatória (para garantir reprodutibilidade)
np.random.seed(42)

## Tamanho da sequência
n = int(1e4)

## Quantidade de sequências independentes
m = int(1e2)

## Parâmetros da simulação
mu = 0 # média populacional
sigma2 = 1 # variância populacional
nu = 1 # graus de liberdade

## Declarando objetos
vY = np.zeros((n, m)) # v.a.'s
vY_bar = np.zeros((n, m)) # médias

## Loop principal
for j in np.arange(0, m, 1):
    for i in np.arange(0, n, 1):
        vY[i, j] = np.random.standard_t(df = nu) 
        vY_bar[i, j] = vY[0:i+1, j].mean()
```


```python
## Vizinhança para o gráfico de convergência (experimente mudar esse valor!)
eps = 0.30

## Gráfico da convergência
plt.figure(figsize = (12, 6))
plt.plot(vY_bar)
plt.axhline(mu - eps, color = 'red', linestyle = '--')
plt.axhline(mu, color = 'black', linestyle = '--', label = f'Média populacional (μ = {mu:.2f})')
plt.axhline(mu + eps, color = 'red', linestyle = '--')
plt.title('Demonstração da WLLN', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Sequências de Médias (Amostrais)', fontsize = 12)
plt.ylim(-2.0, 2.0)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_22_0.png)
    



```python
## Calculando EQMs e EAMs
vMSE = np.zeros(n) # EQMs
vMAE = np.zeros(n) # EAMs

## Loop principal
for i in np.arange(0, n, 1):
    vMSE[i] = ((vY_bar[i, :] - mu)**2).mean()
    vMAE[i] = (abs(vY_bar[i, :] - mu)).mean()
```


```python
## Gráfico da convergência do EQM e EAM
plt.figure(figsize = (12, 6))
plt.plot(vMSE, color = 'blue', label = 'MSE')
plt.plot(vMAE, color = 'red', label = 'MAE')
plt.axhline(0.00, color = 'black', linestyle = '--')
plt.title('Risco como função de n', fontsize=14)
plt.xlabel('Tamanho amostral (n)', fontsize=12)
plt.ylabel('Estimativas dos riscos', fontsize = 12)
plt.ylim(-0.01, 100)
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.show()
```


    
![png](T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_files/T%C3%B3pico%2003%20%E2%80%93%20Estima%C3%A7%C3%A3o%20Pontual%20%28Risco%29_24_0.png)
    

