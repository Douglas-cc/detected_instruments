import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt                
from sklearn.preprocessing import LabelEncoder
from src.wrapped import Wrapped
from src.analysesV02 import Analytics 
from src.trainV02 import TrainModels
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


tm = TrainModels()
wp = Wrapped(
    '../data/row/',
    '../data/processed/',
    '../data/files/'
)

# dataframe
df = wp.load_data('df_instrumentos_features_selecionadas').drop(columns=['file_name'])
df = df.query("instrumento != 'voice' & instrumento != 'synthesizer'")

# dataframes por instrumentos
inst_corda    = ["cello", "guitar", "violin", "bass", "banjo", "mandolin", "ukulele"]
inst_percusao = ["mallet_percussion", "drums", "cymbals"]
inst_sopro    = ["clarinet", "trombone", "flute", "trumpet", "saxophone"]
inst_aerofone = ["accordion", "organ", "piano"] 

df_inst_aerofone = df[df['instrumento'].isin(inst_aerofone)]
df_inst_sopro    = df[df['instrumento'].isin(inst_sopro)]
df_inst_corda    = df[df['instrumento'].isin(inst_corda)]
df_inst_percusao = df[df['instrumento'].isin(inst_percusao)]

# label encoding 
le = LabelEncoder()
df_inst_aerofone['labels'] = le.fit_transform(df_inst_aerofone.instrumento)
df_inst_sopro['labels']    = le.fit_transform(df_inst_sopro.instrumento)
df_inst_corda['labels']    = le.fit_transform(df_inst_corda.instrumento)
df_inst_percusao['labels'] = le.fit_transform(df_inst_percusao.instrumento)


# Hist Gradiente Boosting para Sopro e Percusao
parametros_histGB = {
    "min_samples_leaf": Integer(5, 20),
    "max_depth": Integer(6, 20),
    "loss": Categorical(['log_loss','auto','categorical_crossentropy']), 
    "max_bins": Integer(100, 250)
}

tm.train_tunning_hyperparameters(
    dataframe=df_inst_percusao, 
    model=HistGradientBoostingClassifier(), 
    parameters=parametros_histGB, 
    filename="resultados_parametros_percusao_histGB"
)

# Random Forest 
parametros_rf = {
    "criterion": Categorical(['gini','entropy']),
    "max_depth": Integer(6, 20),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(2, 10),
    "max_features": Categorical(['auto', 'sqrt','log2']), 
    "bootstrap": Categorical([True, False]),
    "n_estimators": Integer(100, 500)
}

tm.train_tunning_hyperparameters(
    dataframe=df_inst_aerofone, 
    model=RandomForestClassifier(), 
    parameters=parametros_rf, 
    filename="resultados_parametros_aerofone_random_forest"
)

tm.train_tunning_hyperparameters(
    dataframe=df_inst_sopro, 
    model=RandomForestClassifier(), 
    parameters=parametros_rf, 
    filename="resultados_parametros_sopro_random_forest"
)

tm.train_tunning_hyperparameters(
    dataframe=df_inst_percusao, 
    model=RandomForestClassifier(), 
    parameters=parametros_rf, 
    filename="resultados_parametros_percusao_random_forest"
)


tm.train_tunning_hyperparameters(
    dataframe=df_inst_corda, 
    model=RandomForestClassifier(), 
    parameters=parametros_rf, 
    filename="resultados_parametros_corda_random_forest"
)