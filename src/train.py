import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.utils import  class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import accuracy_score
from src.analysesV02 import Analytics


ac = Analytics()

def cross_validate_balancead(k, model, X, y, oversampling=False, weight=False):
    kfold =  StratifiedKFold(n_splits=k) 
    accuracy_split = []
    
    # interando sobre os splits
    for idx, (idx_train, idx_validate) in enumerate(kfold.split(X, y)):
        X_split_train = X.iloc[idx_train, :]
        y_split_train = y.iloc[idx_train, :]
        
        if oversampling:
            sm = SMOTE(random_state=42)
            X_split_train, y_split_train = sm.fit_resample(X_split_train, y_split_train)
            
        if weight:
            weights = class_weight.compute_class_weight(
                class_weight = 'balanced',
                classes = np.unique(y_split_train),
                y = y_split_train.values.reshape(-1)
            )
                    
        # com os dados balanceados SÓ NO TREINO, vamos treinar 
        model.fit(X_split_train, y_split_train.values.flatten())
        
        # splist para validação
        X_split_validate = X.iloc[idx_validate, :]
        y_split_validate = y.iloc[idx_validate, :]
        
        # validacao SEM oversampling, amostra do mundo real com dados desbalanceados
        predictions_val = model.predict(X_split_validate)
        accuracy = accuracy_score(y_split_validate, predictions_val)
        accuracy_split.append(accuracy)
        print(f'Acuracia do modelo {model} do Fold {idx}: {accuracy}')
    return np.mean(accuracy_split)
        

def train_feature_combination(model, df, list_features, size_comb):
    count = 0
    dic_result= defaultdict(list)
    comb_features = np.array(list(combinations(list_features, size_comb)))
    
    for i in comb_features:
        count = count + 1
        X = df.iloc[:,i]
        print(f'Teste {count} -> features Selecionada para o treino: {X.columns}')
        

        accuracy = cross_validate_balancead(k=5,  model=model, X=X, y=df['labels'].to_frame())
        print(f'Accuracy {accuracy} do teste -> {count}')

        if accuracy >= 0.7:
            dic_result['features'].append(X.columns)
            dic_result['accuracy'].append(accuracy)
            # break

    return dic_result            


def train_models(X, y):
    # X = df_train.drop(columns=['labels'])
    # y = df_train['labels'].to_frame()
    models = np.array([
        GaussianNB(),
        KNeighborsClassifier(), 
        DecisionTreeClassifier(), 
        RandomForestClassifier(), 
        HistGradientBoostingClassifier(),
        LGBMClassifier(),
        MLPClassifier(),
        XGBClassifier(),
        SVC(),
    ])

    acuracy_models = [cross_validate_balancead(k=5, model=model, X=X,  y=y) for model in models]

    dict_results = {
        'Naive Bayes': acuracy_models[0],
        'KNN': acuracy_models[1],
        'Arvore de Decisão': acuracy_models[2],
        'Floresta Aleatoria': acuracy_models[3],
        'HistGradientBoosting': acuracy_models[4],
        'LIGHTGBM': acuracy_models[5],
        'MLP': acuracy_models[6],
        'XGB': acuracy_models[7],
        'SVC': acuracy_models[8],
    }
    return dict_results


def train_tunning_hyperparameters(dataframe, model, parameters, filename,log=False, cv=5):    
    parameters_knn = ac.tunning_hyperparameters_knn(dataframe=dataframe, log=log)
    bayes_search = BayesSearchCV(
        model,
        parameters,
        n_iter=32,
        n_jobs=-1,
        cv=cv,
        scoring='accuracy'
    )

    output = []
    for i in range(len(parameters_knn)):
        print(f'interação {i} - Metric: {parameters_knn[i]["Metric"]}, Algoritmo: {parameters_knn[i]["algorithm"]}, neighbor: {parameters_knn[i]["neighbors"]}')
    
        new_df = ac.show_outilers(dataframe=dataframe, pred=parameters_knn[i]['outilers'])

        X = new_df.drop(columns=["instrumento", "labels"])
        y = new_df["labels"]

        bayes_search.fit(X, y)

        aux = {
            "parametos_modelo_outliers": {
                "Metric":parameters_knn[i]["Metric"],
                "Algorithm":parameters_knn[i]["algorithm"],
                "Neighbors":parameters_knn[i]["neighbors"]
            },
            "parametos_modelo":bayes_search.best_params_,
            "acuracy":bayes_search.best_score_ * 100,
        }

        print(f'interação {i}', aux)
        output.append(aux)
        
    pickle.dump(output, open(wp.directory_row+filename, 'wb'))
    return output    