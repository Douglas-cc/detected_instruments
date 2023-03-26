# Sobre 



# Parametros PYOD
## Decission Function:

**decision_function():** Este método retorna o score de decisão para cada amostra. O score de decisão é uma medida de quão provável é que uma determinada amostra seja um outlier. O valor do score de decisão depende do algoritmo específico utilizado. Para alguns algoritmos, um valor maior indica uma maior probabilidade de ser um outlier, enquanto para outros, um valor menor indica uma maior probabilidade de ser um outlier.


## Interpretando o score para o caso do KNN

decision_function() do algoritmo KNN retorna a distância média das k amostras mais próximas para cada amostra no conjunto de dados. O valor retornado pelo método decision_function() pode ser interpretado como o grau de anomalia da amostra, em que valores mais altos indicam maior grau de anomalia.

A interpretação do valor do score de decisão para o KNN pode ser feita considerando os seguintes pontos:

Quanto maior o valor do score de decisão, maior a distância média das k amostras mais próximas. Isso significa que a amostra está mais distante das amostras vizinhas e, portanto, é menos semelhante a elas. Em outras palavras, a amostra é mais anômala em relação ao seu entorno.

Para escolher um valor de corte para classificar as amostras como outlier ou normal, é importante levar em consideração a distribuição dos valores de score de decisão. Normalmente, os valores de score de decisão seguem uma distribuição normal ou uma distribuição de cauda longa. Um valor de corte adequado pode ser escolhido com base na média e no desvio padrão da distribuição dos valores de score de decisão.

O valor do score de decisão não fornece informações sobre a classe de outlier à qual a amostra pertence, apenas indica a probabilidade de ser um outlier. Portanto, é importante avaliar visualmente os resultados do modelo e analisar as amostras identificadas como outliers para determinar o tipo de anomalia presente nos dados.

Em resumo, o score de decisão retornado pelo KNN indica a distância média das k amostras mais próximas e pode ser interpretado como uma medida de quão anômala a amostra é em relação ao seu entorno. Um valor de corte adequado deve ser escolhido com base na distribuição dos valores de score de decisão.


- Decission Scores:

**decision_scores()** é um dos atributos mais importantes da biblioteca PyOD. Ele é usado para retornar o score de decisão para cada amostra no conjunto de dados. O score de decisão é uma medida de quão provável é que uma determinada amostra seja um outlier, e é usado para classificar cada amostra como outlier ou normal.

O valor do score de decisão depende do algoritmo específico utilizado. Para alguns algoritmos, um valor maior indica uma maior probabilidade de ser um outlier, enquanto para outros, um valor menor indica uma maior probabilidade de ser um outlier. Por exemplo, no algoritmo KNN, o score de decisão é a distância média das k amostras mais próximas, enquanto no algoritmo ABOD, o score de decisão é a média das distâncias de Mahalanobis de cada amostra em relação às outras amostras.

O atributo decision_scores_ é útil porque fornece uma maneira de ajustar o nível de contaminação (isto é, a proporção de outliers esperados) para um determinado conjunto de dados. A partir do score de decisão, pode-se definir um valor de corte para classificar as amostras como outliers ou normais. O valor de corte pode ser escolhido manualmente ou usando técnicas estatísticas ou de aprendizado de máquina para determinar automaticamente o melhor valor.

Em resumo, o atributo decision_scores_ é usado para retornar o score de decisão para cada amostra em um conjunto de dados. O valor do score de decisão depende do algoritmo específico utilizado e é usado para classificar cada amostra como outlier ou normal, geralmente usando um valor de corte definido pelo usuário ou por uma técnica automática.

- Predict:

**predict():** Este método retorna uma matriz binária indicando se cada amostra é um outlier ou não. Os outliers são marcados como 1 e as amostras normais são marcadas como 0. O valor de corte para determinar se uma amostra é um outlier ou não depende do algoritmo específico utilizado e do nível de contaminação definido.

- Labels:

**labels_:** Este atributo retorna uma matriz binária indicando se cada amostra é um outlier ou não, assim como o método predict(). No entanto, o atributo labels_ é definido apenas para alguns algoritmos que suportam a detecção de outliers de várias classes. Nesses casos, labels_ é uma matriz em que cada linha corresponde a uma amostra e cada coluna corresponde a uma classe. O valor da célula (i,j) indica se a amostra i pertence à classe j (0 para amostras normais e 1 para outliers)


## Principal diferenaça entre Labels e Predict:

O método predict() e o atributo labels_ são usados para identificar os outliers em um conjunto de dados. No entanto, há uma diferença importante entre eles:

A principal diferença entre esses dois métodos é que o predict() é usado para detectar outliers em um problema de detecção binária de outliers (onde apenas uma classe de outliers é considerada), enquanto labels_ é usado para detectar outliers em um problema de detecção multiclasse de outliers (onde existem várias classes de outliers).

Por exemplo, se estivermos trabalhando com um problema de detecção binária de outliers (por exemplo, detecção de transações fraudulentas em um conjunto de dados financeiros), o método predict() seria o mais adequado para identificar os outliers. Por outro lado, se estivermos trabalhando com um problema de detecção multiclasse de outliers (por exemplo, detecção de anomalias em um sistema de produção com várias falhas diferentes), o atributo labels_ seria mais útil para identificar cada tipo de falha separadamente.

Em resumo, a principal diferença entre predict() e labels_ é que o primeiro é usado em problemas de detecção binária de outliers e o segundo é usado em problemas de detecção multiclasse de outliers.

# Instalando as dependecias 

- Instalar o arquivo python
```bash
python setup.py develop
```

- Instalando versão python com pyenv 
```bash
pyenv install 3.10.0
```

- Definindo versão local do projeto
```bash
pyenv local 3.10.0
```

- Criando um ambiente virtual com versão de cima do Python
```bash 
poetry env use 3.10
```

- Iniciar o ambiente virtual
```bash 
poetry shell
```