import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.utils import  class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score



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
