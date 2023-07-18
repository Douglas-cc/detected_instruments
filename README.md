# Sobre 
O projeto de classificação de instrumentos musicais utiliza técnicas avançadas de machine learning e MIR (Music Information Retrieval) para treinar modelos de classificação capazes de identificar e categorizar diferentes instrumentos musicais a partir de amostras de áudio.

A classificação de instrumentos musicais é uma tarefa desafiadora devido à variedade de timbres e características sonoras únicas de cada instrumento. No entanto, com o avanço da tecnologia, tornou-se possível desenvolver algoritmos e modelos capazes de reconhecer e diferenciar esses sons com uma precisão cada vez maior.

O processo de treinamento desses modelos envolve a utilização de técnicas de machine learning, em que uma grande quantidade de dados de áudio é coletada e rotulada com as respectivas classes de instrumentos. Esses dados são então usados para treinar o modelo, que aprenderá a reconhecer padrões e características específicas de cada instrumento.

Além do machine learning, técnicas de MIR também desempenham um papel fundamental nesse projeto. O MIR é uma área de pesquisa que se concentra na extração de informações musicais a partir de sinais de áudio. No contexto da classificação de instrumentos musicais, técnicas de MIR são aplicadas para extrair características relevantes dos sinais de áudio, como espectrogramas, frequências fundamentais e características de envoltória.

Combinando técnicas de machine learning e MIR, os modelos de classificação de instrumentos musicais são capazes de aprender a distinguir entre diferentes instrumentos com base em características acústicas específicas. Esses modelos podem ser aplicados em diversas áreas, como análise musical, reconhecimento automático de músicas e até mesmo na criação de instrumentos virtuais mais realistas.

O projeto de classificação de instrumentos musicais tem o potencial de contribuir para o desenvolvimento de tecnologias musicais avançadas, possibilitando a criação de ferramentas e sistemas mais inteligentes para músicos, produtores e entusiastas da música. Ao utilizar técnicas de machine learning e MIR, esse projeto abre novas possibilidades para a pesquisa e o avanço da compreensão e interação com a música.

# Metodologia

### Criação da base de dados 

### Seleção das melhores features

### Remoção de outilers

### Seleção de Melhores parametros

### Treinamento de Modelos


# Resultados Finais


# Estrutura do Projeto

```
detected-instruments  # Project folder
├── app               # template API deploy model
├── data              # Local project data
├── docs              # Documentation
├── notebooks         # Exploratory Jupyter notebooks 
├── pyproject.toml    # Identifies the project root
├── README.md         # README.md explaining your project
├── setup.py          # Configuration project
└── src               # Source code 
```
# Instalando as dependecias 

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
- Instalar bibliotecas python .toml
```bash 
poetry install
```

- Instalar o arquivo python
```bash
python setup.py develop
```
