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
# inst_corda    = ["cello", "guitar", "violin", "bass", "banjo", "mandolin", "ukulele"]
# inst_percusao = ["mallet_percussion", "drums", "cymbals"]
inst_sopro    = ["clarinet", "trombone", "flute", "trumpet", "saxophone"]
# inst_aerofone = ["accordion", "organ", "piano"] 

# df_inst_aerofone = df[df['instrumento'].isin(inst_aerofone)]
df_inst_sopro    = df[df['instrumento'].isin(inst_sopro)]
# df_inst_corda    = df[df['instrumento'].isin(inst_corda)]
# df_inst_percusao = df[df['instrumento'].isin(inst_percusao)]

# label encoding 
le = LabelEncoder()
# df_inst_aerofone['labels'] = le.fit_transform(df_inst_aerofone.instrumento)
df_inst_sopro['labels']    = le.fit_transform(df_inst_sopro.instrumento)
#df_inst_corda['labels']    = le.fit_transform(df_inst_corda.instrumento)
# df_inst_percusao['labels'] = le.fit_transform(df_inst_percusao.instrumento)


# Tunning de Hiperparametros
parametros_xgb = {
    "eta": Real(0.01, 0.2),
    "max_depth": Integer(6, 20),
    "gamma":  Integer(0, 10),
    "learning_rate": Real(0,1),
    "subsample": Real(0.5, 1)
}

tm.train_tunning_hyperparameters(
    dataframe=df_inst_sopro, 
    model=XGBClassifier(),
    parameters=parametros_xgb, 
    filename="resultados_parametros_sopro_XGBoost"
)

parametros_histGB = {
    "min_samples_leaf": Integer(5, 20),
    "max_depth": Integer(6, 20),
    "loss": Categorical(['log_loss','auto','categorical_crossentropy']), 
    "max_bins": Integer(100, 250)
}

tm.train_tunning_hyperparameters(
    dataframe=df_inst_sopro, 
    model=HistGradientBoostingClassifier(), 
    parameters=parametros_histGB, 
    filename="resultados_parametros_sopro_histGB"
)