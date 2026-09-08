---
layout: default
title: "Tópico 10 – Testes de Hipóteses II"
parent: "Aulas"
nav_order: 10
---
# Tópico 10 – Testes de Hipóteses II [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II/images/colag_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 6.5%;">](https://colab.research.google.com/github/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II.ipynb) [<img src="https://raw.githubusercontent.com/urielmoreirasilva/ICE072/main/aulas/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II/images/github_logo.svg" style="float: right; margin-right: 0%; vertical-align: middle; width: 3.25%;">](https://github.com/urielmoreirasilva/ICE072/blob/main/aulas/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II.ipynb)




### MATERIAL EM CONSTRUÇÃO [!]

Material adaptado do [DSC10 (UCSD)](https://dsc10.com/) por [Flavio Figueiredo (DCC-UFMG)](https://flaviovdf.io/fcd/) e [Uriel Silva (DEST-UFMG)](https://urielmoreirasilva.github.io)


```python
## Imports para esse tópico
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns
```


```python
## Parâmetros do matplotlib
# Seaborn
plt.style.use('seaborn-v0_8-colorblind')
# Outros
plt.rcParams['figure.figsize']  = (16, 10)
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 4
```


```python
## Contexto do matplotlib
plt.ion()
```




    <contextlib.ExitStack at 0x1b4a0e58830>




```python
## Remover eixos dos plots do matplotlib
def despine(ax = None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
```


```python
## Fixando semente aleatória
np.random.seed(42) # vide Guia do Mochileiro das Galáxias
```

## Exemplo 1: Júri no Alabama

No início dos anos 1960, no Condado de Talladega, no Alabama, um negro chamado Robert Swain foi condenado por  à morte por estuprar uma mulher branca. Ele recorreu da sentença, citando entre outros fatores o júri todo branco. Na época, apenas homens com 21 anos ou mais eram autorizados a servir em júris no condado de Talladega. No condado, 26% dos jurados elegíveis eram negros. No júri final, havia apenas 8 negros entre os 100 selecionados para o painel de jurados no julgamento de Swain.

Nossa pergunta: **Qual é a probabilidade de serem selecionados 8 indíviduos negros em 100?**

### Método #1: Simulando proporções de uma população qualquer

Para nos ajudar com este exemplo, o código abaixo amostra proporções de uma população. O mesmo gera 10,000 amostras ($n$) de uma população de tamanho `pop_size`. Tais amostras são geradas *sem reposição*. Além do mais, o código assume que:

1. [`0`, `pop_size * prop`) pertencem a um grupo.
1. [`pop_size * prop`, `pop_size`) pertencem a outro grupo.

Ou seja, em uma população de 10 (`pop_size`) pessoas, caso `prop = 0.2`, então a população é algo do tipo:

> __[G1, G1, G2, G2, G2, G2, G2, G2, G2, G2]__

Nossa ideia é  tentar responder: **Ao realizar amostras uniformes da população acima, quantas pessoas do tipo G1 e do tipo G2 são amostradas?**. 

Para isto, realizamos $M = 10{,}000$ amostras *sem reposição*.


```python
def sample_proportions(pop_size, prop, M = 10000):
    '''
    Amostra proporções de uma população.
    
    Parâmetros
    ----------
    pop_size: int, tamanho da população
    prop: double, entre 0 e 1
    n: int, número de amostras
    '''
    assert(prop >= 0)
    assert(prop <= 1)
    
    grupo = pop_size * prop # tamanho do grupo a ser amostrado
    resultados = np.zeros(M) # array vazio para receber as proporções
    for i in range(M):
        sample = np.random.randint(0, pop_size, 100) # amostra 100 elementos em [0, pop_size)
        resultados[i] = np.sum(sample < grupo) # conta o *número* de *amostras* menores que o tamanho do grupo
    return resultados
```


```python
?np.random.randint
```


    [31mSignature:[39m np.random.randint(low, high=[38;5;28;01mNone[39;00m, size=[38;5;28;01mNone[39;00m, dtype=<[38;5;28;01mclass[39;00m [33m'int'[39m>)
    [31mDocstring:[39m
    randint(low, high=None, size=None, dtype=int)
    
    Return random integers from `low` (inclusive) to `high` (exclusive).
    
    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).
    
    .. note::
        New code should use the `~numpy.random.Generator.integers`
        method of a `~numpy.random.Generator` instance instead;
        please see the :ref:`random-quick-start`.
    
    Parameters
    ----------
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int or array-like of ints, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
        If array-like, must contain integer values
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. Byteorder must be native.
        The default value is long.
    
        .. warning::
          This function defaults to the C-long dtype, which is 32bit on windows
          and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones).
          Since NumPy 2.0, NumPy's default integer is 32bit on 32bit platforms
          and 64bit on 64bit platforms.  Which corresponds to `np.intp`.
          (`dtype=int` is not the same as in most NumPy functions.)
    
    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.
    
    See Also
    --------
    random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted.
    random.Generator.integers: which should be used for new code.
    
    Examples
    --------
    >>> np.random.randint(2, size=10)
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
    >>> np.random.randint(1, size=10)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    Generate a 2 x 4 array of ints between 0 and 4, inclusive:
    
    >>> np.random.randint(5, size=(2, 4))
    array([[4, 0, 2, 1], # random
           [3, 2, 2, 0]])
    
    Generate a 1 x 3 array with 3 different upper bounds
    
    >>> np.random.randint(1, [3, 5, 10])
    array([2, 2, 9]) # random
    
    Generate a 1 by 3 array with 3 different lower bounds
    
    >>> np.random.randint([1, 5, 7], 10)
    array([9, 8, 7]) # random
    
    Generate a 2 by 4 array using broadcasting with dtype of uint8
    
    >>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
    array([[ 8,  6,  9,  7], # random
           [ 1, 16,  9, 12]], dtype=uint8)
    [31mType:[39m      method



```python
## Fazendo uma das iterações acima na mão

# declarações
pop_size = 200
prop = 0.08

# grupo
grupo = pop_size * prop
print('grupo =', grupo)

# amostragem
sample = np.random.randint(0, pop_size, 100) # amostra *sem reposição*!
print('sample = ', sample)

# saída
print('resultados[i] = ', np.sum(sample < grupo))

# saída (em proporção)
print('resultados[i] = ', np.sum(sample < grupo)/100)
```

    grupo = 16.0
    sample =  [102 179  92  14 106  71 188  20 102 121  74  87 116  99 103 151 130 149
      52   1  87 157  37 129 191 187  20 160  57  21  88  48  58 169 187  14
     189 189 174 189  50 107  54  63 130  50 134  20  72 166  17 131  88  59
      13   8  89  52 129  83  91 110 187 198 171   7 174  34  80 163  49 103
     131   1 133  53 105   3  53 190 145  43 161 189  13  94  47  14 199 189
      39  81 110  52  23 153 187 123  40 156]
    resultados[i] =  10
    resultados[i] =  0.1
    

Vamos ver agora a distribuição empírica de $M = 10{,}000$ amostras da cidade de Talladega. Inicialmente, vamos assumir que a cidade tem uma população de $ n = 100{,}000$ habitantes*.

*O número de habitantes não importa muito para o exemplo, uma vez que estamos apenas gerando amostras aleatórias da população. É importante apenas que $n$ seja grande o suficiente para que o comportamento dos teoremas limite que estamos utilizando como garantia teórica sejam válidos.


```python
## Distribuição empírica de M amostras da cidade de Talladega
np.random.seed(42)
n = int(1e5) # para formatação correta no gráfico
proporcoes = sample_proportions(pop_size = n, prop = 0.26)
bins = np.linspace(1, 100, 100) + 0.5
plt.hist(proporcoes, bins = bins, edgecolor='k')
plt.xlim(0, 52)
plt.ylabel(f'Número de Amostras de Tamanho n = {n}')
plt.xlabel('Número no Grupo')
plt.plot([8], [0], 'ro', ms=15)
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_13_0.png)
    


Com 5\% de probabilidade, pelo menos 19 pessoas negras são amostradas em 100...


```python
np.percentile(proporcoes, 5)
```




    np.float64(19.0)



... e o valor "8" nem aparece na distribuição da população original!


```python
np.count_nonzero(proporcoes < 8) 
```




    np.int64(0)




```python
p_value = np.count_nonzero(proporcoes < 8) / len(proporcoes)
p_value
```




    np.float64(0.0)



### Método #2: Simulando sob $H_0$


```python
## Simulando (sob H_0) a distribuição de T
# ----

## Declarações
np.random.seed(42)
M = int(1e4) # necessário para uso no loop
n = int(1e5) # tamanho da população

## Loop principal
results = np.array([])
for i in np.arange(M):
    result = np.random.binomial(n, 0.26)*100/n
    results = np.append(results, result)
results
```




    array([25.863, 26.112, 26.015, ..., 25.597, 26.028, 26.118],
          shape=(10000,))




```python
(pd.DataFrame({"results" : results})
 .plot(kind = 'hist', bins = 50,
       density = True, ec = 'w', figsize=(10, 5),
       title='Distribuição Empírica [...]'))
# plt.axvline(8, color = 'black', linewidth = 4, label = 'T_obs = 0.08')
plt.legend()
plt.ylabel("Densidade");
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_21_0.png)
    



```python
np.percentile(results, 5)
```




    np.float64(25.766)




```python
np.count_nonzero(results < 0.08) 
```




    np.int64(0)




```python
p_value = np.count_nonzero(results < 0.08) / len(results)
p_value
```




    np.float64(0.0)



### Método #3: Bootstrap!


```python
## Aproximando a distribuição de T via boostrap
# ----

## Declarações
np.random.seed(42)
population = pd.DataFrame({'negros' : np.append(np.ones(8000), np.zeros(92000))}) # "amostra" da cidade
M = int(1e4) # necessário para uso no loop
n = int(1e5) # tamanho da população

## Loop principal
results_boot = np.array([])
for i in np.arange(M):
    result_boot = np.sum(population.sample(n, replace = True), axis = 0)/n
    results_boot = np.append(results_boot, result_boot)
```


```python
(pd.DataFrame({"results_boot" : results_boot})
 .plot(kind = 'hist', bins = 50,
       density = True, ec = 'w', figsize=(10, 5),
       title='Distribuição Empírica [...]'))
# plt.axvline(0.26, color = 'black', linewidth = 4, label = 'p_0 = 0.26')
plt.legend()
plt.ylabel("Densidade");
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_27_0.png)
    



```python
L = np.percentile(results_boot, 2.5)
U = np.percentile(results_boot, 97.5)
[L, U]
```




    [np.float64(0.07832), np.float64(0.08166)]



## Exemplo 2: Um outro Júri

"In 2010, the American Civil Liberties Union (ACLU) of Northern California presented a report on jury selection in Alameda County, California. The report concluded that certain ethnic groups are underrepresented among jury panelists in Alameda County, and suggested some reforms of the process by which eligible jurors are assigned to panels."

"1453 people reported for jury duty in total (we will call them "panelists")".

Aqui temos um outro exemplo de júri. Neste caso, temos diferentes grupos raciais. 


```python
## Júri alternativo
idx = ['Asian', 'Black', 'Latino', 'White', 'Other']
df = pd.DataFrame(index = idx)
df['pop'] = [0.15, 0.18, 0.12, 0.54, 0.01]
df['sample'] = [0.26, 0.08, 0.08, 0.54, 0.04]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop</th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asian</th>
      <td>0.15</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>0.18</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>Latino</th>
      <td>0.12</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>White</th>
      <td>0.54</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>0.01</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>



Agora, com base na distribuição amostral acima, como podemos inferir algo sobre a composição racial da população?

Na prática, seria bem difícil trabalhar com 5 estatísticas de teste...


```python
## Visualizando as proporções na amostra e na população
df.plot.bar()
plt.ylabel('Propopção')
plt.ylabel('Grupo')
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_32_0.png)
    


### Método #1: Simulando proporções de uma população qualquer

Para operacionalizar o Método #1 aqui, podemos usar a **distância de variação total**, ou __total variation distance (TVD)__!

Para distribuições discretas (o caso mais comum aqui são as *categóricas*, quase sempre um caso particular da *distribuição Multinomial*), a TVD toma a forma
$$TVD(\boldsymbol{p}, \boldsymbol{q}) := \frac{1}{2} \sum_{j=1}^m |p_j - q_j|,$$
onde $\boldsymbol{p}$ e $\boldsymbol{q}$ aqui são vetores de probabilidades (ou proporções amostrais). 

A TVD é de fato uma **distância**, então *quanto maior o valor da TVD, maior a diferença entre as proporções dos dois vetores*!

Naturalmente, quando $\boldsymbol{p} = \boldsymbol{q}$, temos $TVD(\boldsymbol{p}, \boldsymbol{q}) = 0$.


```python
def TVD(p, q):
    '''
    Computa a TVD de dois vetores de probabilidades/proporções, p e q
    
    Parâmetros
    ----------
    p: vetor de probabilidades de tamanho n
    q: vetor de probabilidades de tamanho n
    '''
    return np.sum(np.abs(p - q)) / 2
```

Vamos comparar então as proporções $\hat{\boldsymbol{p}}$ na nossa amostra com as proporções obtidas com base em uma _amostra aleatória_ da população!

Primeiramente, começamos amostrando um grupo de cada vez:


```python
np.random.seed(42) # semente aleatória
N = 1453 # tamanho da população
uma_amostra = [] # array vazio para receber as proporções
for g in df.index: # uma amostragem (M = 10.000 replicações) para cada grupo
    p = df.loc[g]['pop'] # proporção na população
    s = sample_proportions(N, p, 1)[0] # coagindo para double
    uma_amostra.append(s/100) # transformando em proporções
```


```python
sample_proportions(N, p, 1)[0]
```




    np.float64(1.0)




```python
uma_amostra
```




    [np.float64(0.16),
     np.float64(0.1),
     np.float64(0.15),
     np.float64(0.54),
     np.float64(0.01)]




```python
df['single random sample'] = uma_amostra
df.plot.bar()
plt.ylabel('Propopção')
plt.ylabel('Grupo')
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_40_0.png)
    


Agora, vamos comparar a TVD nessa amostra aleatória com a TVD na amostra dos panelistas: 


```python
TVD(df['single random sample'], df['pop'])
```




    np.float64(0.06)




```python
TVD(df['sample'], df['pop'])
```




    np.float64(0.14)



De maneira análoga ao que fizemos acima, aqui novamente tomamos $M = 10{,}000$ amostras e comparamos suas TVDs!

O código abaixo armazena o resultado de cada amostra em uma _linha_ de uma matriz.


```python
## Distribuição amostral das TVDs
np.random.seed(42)
N = 1453
A = np.zeros(shape = (10000, len(df.index))) # len(df.index) = 5 aqui
for i, g in enumerate(df.index): # o enumerate é necessário para fazer o for na ordem certa do segundo iterador
    p = df.loc[g]['pop']
    A[:, i] = sample_proportions(N, p) / 100
```


```python
A
```




    array([[0.16, 0.21, 0.16, 0.55, 0.  ],
           [0.08, 0.16, 0.13, 0.54, 0.01],
           [0.19, 0.2 , 0.15, 0.61, 0.  ],
           ...,
           [0.14, 0.13, 0.1 , 0.58, 0.  ],
           [0.09, 0.09, 0.12, 0.52, 0.  ],
           [0.07, 0.19, 0.16, 0.63, 0.  ]], shape=(10000, 5))




```python
all_distances = []
for i in range(A.shape[0]):
    all_distances.append(TVD(df['pop'], A[i]))
```


```python
all_distances[0:10] # array com M linhas
```




    [np.float64(0.05000000000000001),
     np.float64(0.049999999999999996),
     np.float64(0.08499999999999999),
     np.float64(0.10000000000000002),
     np.float64(0.09999999999999998),
     np.float64(0.05499999999999999),
     np.float64(0.06999999999999998),
     np.float64(0.045000000000000005),
     np.float64(0.08000000000000002),
     np.float64(0.07499999999999997)]




```python
plt.hist(all_distances, bins = 30, edgecolor='k')
plt.ylabel(f'Numero de Amostras de Tamanho N = {N}')
plt.xlabel('Total Variation Distance')
plt.plot([0.14], [0], 'ro', ms=15)
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_49_0.png)
    



```python
np.percentile(all_distances, 95)
```




    np.float64(0.11)




```python
np.count_nonzero(np.array(all_distances) > 0.14) # precisamos converter para array para vetorizar, porque all_distances é uma lista
```




    np.int64(43)




```python
p_value = np.count_nonzero(np.array(all_distances) > 0.14) / len(all_distances)
p_value
```




    np.float64(0.0043)



### Método #2. Simulando sob $H_0$

Nesse caso, o problema é um pouco mais complicado do que o de simular apenas uma contagem/proporção sob $H_0$ (mas não muito!)

Quando temos apenas um grupo (ou dois, um de "sucessos" e o outro de "fracassos"), basta simularmos
$$T(\boldsymbol{Y}) \overset{H_0}{\sim} Binomial(n, p_0),$$
onde $T(\boldsymbol{Y}) := n \cdot \hat{p}$.

Porém, se tivermos $m$ grupos, $m \geq 3$, na verdade sob $H_0$ temos que
$$\boldsymbol{G}(\boldsymbol{Y}) \overset{H_0}{\sim} Multinomial(n, \boldsymbol{p}),$$
onde $\boldsymbol{G}(\boldsymbol{Y}) := (G_1, \ldots, G_m)^T$ é tal que $G_j \equiv G_j(\boldsymbol{Y})$ denota o _número de elementos da amostra $\boldsymbol{Y}$ correspondente ao grupo $j$_, e $\boldsymbol{p} := (p_1, \ldots, p_j)$ denota o vetor das probabilidades/proporções de cada grupo na população.

Dessa forma, embora nesse caso ainda saibamos como simular sob $H_0$ (basta usar a função `np.random.multinomial`), temos o mesmo problema que com o Método #1 anterior: $\boldsymbol{G}(\boldsymbol{Y})$ tem dimensão $m$!

Inicialmente, nossa maneira de resolver esse problema será a mesma: tomaremos nossa estatística de teste como $T(\boldsymbol{Y}) = TVD(\hat{\boldsymbol{p}}, \boldsymbol{p})$.

<u> Nota</u>: Até onde se sabe, sob $H_0$ a TVD não tem nenhuma distribuição conhecida, o que impossibilita a obtenção de uma região de aceitação/crítica exata/analítica. Ainda assim, como o teste resultante é **consistente** (falaremos mais sobre essa definição depois), em geral obtemos resultados muito bons via simulação.  

Primeiramente, vamos recapitular rapidamente como funciona a amostragem de uma distribuição Multinomial através da função `np.random.multinomial`: 


```python
df['pop']
```




    Asian     0.15
    Black     0.18
    Latino    0.12
    White     0.54
    Other     0.01
    Name: pop, dtype: float64




```python
?np.random.multinomial
```


    [31mSignature:[39m np.random.multinomial(n, pvals, size=[38;5;28;01mNone[39;00m)
    [31mDocstring:[39m
    multinomial(n, pvals, size=None)
    
    Draw samples from a multinomial distribution.
    
    The multinomial distribution is a multivariate generalization of the
    binomial distribution.  Take an experiment with one of ``p``
    possible outcomes.  An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6.  Each sample drawn from the
    distribution represents `n` such experiments.  Its values,
    ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
    outcome was ``i``.
    
    .. note::
        New code should use the `~numpy.random.Generator.multinomial`
        method of a `~numpy.random.Generator` instance instead;
        please see the :ref:`random-quick-start`.
    
    .. warning::
      This function defaults to the C-long dtype, which is 32bit on windows
      and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones).
      Since NumPy 2.0, NumPy's default integer is 32bit on 32bit platforms
      and 64bit on 64bit platforms.
    
    
    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        must sum to 1 (however, the last element is always assumed to
        account for the remaining probability, as long as
        ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    
    Returns
    -------
    out : ndarray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.
    
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.
    
    See Also
    --------
    random.Generator.multinomial: which should be used for new code.
    
    Examples
    --------
    Throw a dice 20 times:
    
    >>> np.random.multinomial(20, [1/6.]*6, size=1)
    array([[4, 1, 7, 5, 2, 1]]) # random
    
    It landed 4 times on 1, once on 2, etc.
    
    Now, throw the dice 20 times, and 20 times again:
    
    >>> np.random.multinomial(20, [1/6.]*6, size=2)
    array([[3, 4, 3, 3, 4, 3], # random
           [2, 4, 3, 4, 0, 7]])
    
    For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
    we threw 2 times 1, 4 times 2, etc.
    
    A loaded die is more likely to land on number 6:
    
    >>> np.random.multinomial(100, [1/7.]*5 + [2/7.])
    array([11, 16, 14, 17, 16, 26]) # random
    
    The probability inputs should be normalized. As an implementation
    detail, the value of the last entry is ignored and assumed to take
    up any leftover probability mass, but this should not be relied on.
    A biased coin which has twice as much weight on one side as on the
    other should be sampled like so:
    
    >>> np.random.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
    array([38, 62]) # random
    
    not like:
    
    >>> np.random.multinomial(100, [1.0, 2.0])  # WRONG
    Traceback (most recent call last):
    ValueError: pvals < 0, pvals > 1 or pvals contains NaNs
    [31mType:[39m      method



```python
np.random.multinomial(100, df['pop'])
```




    array([13, 27, 13, 47,  0], dtype=int32)




```python
## Simulando (sob H_0) a distribuição de T
# ----

## Declarações
np.random.seed(42)
M = int(1e4) # necessário para uso no loop
n = 1453 # tamanho da população

## Loop principal
results = np.zeros(shape = (M, len(df.index))) # len(df.index) = 5 aqui
for i in np.arange(M):
    result = np.random.multinomial(n, df['pop'])
    results[i, :] = result/n
results
```




    array([[0.14384033, 0.18238128, 0.10874054, 0.55402615, 0.0110117 ],
           [0.14108741, 0.20233999, 0.12044047, 0.5285616 , 0.00757054],
           [0.14934618, 0.18651067, 0.1321404 , 0.52374398, 0.00825877],
           ...,
           [0.1651755 , 0.17412251, 0.11218169, 0.53200275, 0.01651755],
           [0.14246387, 0.18513421, 0.11699931, 0.5485203 , 0.00688231],
           [0.14521679, 0.1871989 , 0.11493462, 0.54301445, 0.00963524]],
          shape=(10000, 5))




```python
all_distances = []
for i in range(results.shape[0]):
    all_distances.append(TVD(df['pop'], results[i]))
```


```python
all_distances[0:10] # array com M linhas
```




    [np.float64(0.017419132828630418),
     np.float64(0.02278045423262216),
     np.float64(0.01865106675843087),
     np.float64(0.011624225739848556),
     np.float64(0.02929800412938749),
     np.float64(0.032085340674466646),
     np.float64(0.029986235375086063),
     np.float64(0.03130075705437027),
     np.float64(0.017501720578114246),
     np.float64(0.02030970406056436)]




```python
plt.hist(all_distances, bins = 30, edgecolor='k')
plt.ylabel(f'Numero de Amostras de Tamanho N = {N}')
plt.xlabel('Total Variation Distance')
plt.plot([0.14], [0], 'ro', ms=15)
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_65_0.png)
    



```python
np.percentile(all_distances, 97.5)
```




    np.float64(0.03370956641431521)




```python
np.count_nonzero(np.array(all_distances) > 0.14) # precisamos converter para array para vetorizar, porque all_distances é uma lista
```




    np.int64(0)




```python
p_value = np.count_nonzero(np.array(all_distances) > 0.14) / len(all_distances)
p_value
```




    np.float64(0.0)



### Método #3. Distribuição exata...?

Mencionamos acima que $TVD(\hat{\boldsymbol{p}}, \boldsymbol{p})$ não tem distribuição conhecida sob $H_0$.

Porém, uma outra estatística de teste, conhecida como **estatística $\chi^2$** ("qui-quadrado", ou _chi-squared_) (**de Pearson**) e definida por
$$T(\boldsymbol{Y}) := n \cdot \sum^m_{j=1} \frac{(\hat{p}_j - p_j)^2}{p_j},$$
tem distribuição conhecida sob $H_0$!

Mais especificamente, sob $H_0$,
$$T(\boldsymbol{Y}) \overset{d}{\longrightarrow} \chi^2_{m-1}$$
à medida que $n \rightarrow +\infty$.

Similar à TVD, a estatística $\chi^2$ _pode ser interpretada_ como uma medida de distância entre $\hat{\boldsymbol{p}}$ e $\boldsymbol{p}$, pois é sempre não-negativa, e quanto mais "discrepante" $\hat{\boldsymbol{p}}$ for de $\boldsymbol{p}$, maior será o valor dessa estatística.

Porém, _formalmente_, a estatística $\chi^2$ **não é uma medida de distância** entre $\hat{\boldsymbol{p}}$ e $\boldsymbol{p}$, pois ela *não é simétrica* nesses argumentos!

É importante notar também que, como não existe uma transformação um-para-um (isto é, uma bijeção) entre a TVD e a estatística $\chi^2$, então _os testes feitos com base nessas estatísticas não necessariamente têm que dar os mesmos resultados!_

Apesar disso, como ambos os testes são consistentes, a diferença esperada entre eles vai diminuindo à medida que o tamanho amostral cresce.


```python
def chi2(n, p, q):
    '''
    Calcula a estatística \\chi^2 de dois vetores de probabilidades/proporções, p e q
    
    Parâmetros
    ----------
    p: vetor de probabilidades de tamanho n
    q: vetor de probabilidades de tamanho n
    '''
    return n*np.sum( ((p - q)**2) / q )
```


```python
## estatística chi2 na nossa amostra
T_obs = chi2(N, df['pop'], df['sample'])
T_obs
```




    np.float64(310.9978846153846)




```python
## Simulando (sob H_0) a distribuição de T
# (essa parte é idêntica à anterior)

## Declarações
np.random.seed(42)
M = int(1e4) # necessário para uso no loop
n = 1453 # tamanho da população

## Loop principal
results = np.zeros(shape = (M, len(df.index))) # len(df.index) = 5 aqui
for i in np.arange(M):
    result = np.random.multinomial(n, df['pop'])
    results[i, :] = result/n
results
```




    array([[0.14384033, 0.18238128, 0.10874054, 0.55402615, 0.0110117 ],
           [0.14108741, 0.20233999, 0.12044047, 0.5285616 , 0.00757054],
           [0.14934618, 0.18651067, 0.1321404 , 0.52374398, 0.00825877],
           ...,
           [0.1651755 , 0.17412251, 0.11218169, 0.53200275, 0.01651755],
           [0.14246387, 0.18513421, 0.11699931, 0.5485203 , 0.00688231],
           [0.14521679, 0.1871989 , 0.11493462, 0.54301445, 0.00963524]],
          shape=(10000, 5))




```python
all_distances = []
for i in range(results.shape[0]):
    all_distances.append(chi2(N, df['pop'], results[i]))
```


```python
all_distances[0:10] # array com M linhas
```




    [np.float64(2.773438235819934),
     np.float64(5.896726805039254),
     np.float64(3.2215897309063557),
     np.float64(2.469124945020287),
     np.float64(7.405141250399993),
     np.float64(6.04045306110585),
     np.float64(7.225591895186383),
     np.float64(12.878785139961808),
     np.float64(5.5023426027882945),
     np.float64(5.993592453957197)]




```python
plt.hist(all_distances, bins = 30, edgecolor='k')
plt.ylabel(f'Numero de Amostras de Tamanho N = {N}')
plt.xlabel('Estatística $\\chi^2$ de Pearson')
plt.plot([T_obs], [0], 'ro', ms=15)
despine()
```


    
![png](T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_files/T%C3%B3pico%2010%20%E2%80%93%20Testes%20de%20Hip%C3%B3teses%20II_79_0.png)
    



```python
np.percentile(all_distances, 95)
```




    np.float64(10.136629226947521)




```python
np.count_nonzero(np.array(all_distances) > T_obs) # precisamos converter para array para vetorizar, porque all_distances é uma lista
```




    np.int64(0)




```python
p_value = np.count_nonzero(np.array(all_distances) > T_obs) / len(all_distances)
p_value
```




    np.float64(0.0)



Finalmente, podemos comparar esse resultado também com a distribuição exata de $T(\boldsymbol{Y})$ sob $H_0$!


```python
## Região crítica (unilateral)
ss.chi2.ppf(q = 0.95, df = len(df.index) - 1)
```




    np.float64(9.487729036781154)




```python
## p-valor
1 - ss.chi2.cdf(x = T_obs, df = len(df.index) - 1)
```




    np.float64(0.0)


