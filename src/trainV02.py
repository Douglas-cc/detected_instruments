import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.utils import  class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from src.wrapped import Wrapped
from src.analysesV02 import Analytics


class TrainModels:
    def __init__(self):
        self.count = 0
        self.dic_result= defaultdict(list)
        self.ac = Analytics()
        self.wp = Wrapped(
            '../data/row/',
            '../data/processed/',
            '../data/files/'
        )

    def cross_validate_balancead(self, k, model, X, y, oversampling=False, weight=False):
        kfold =  StratifiedKFold(n_splits=k) 

        # tranformando y de series para dataframe de unidimensão
        y = y.to_frame()

        # arrays resultados
        accuracy_split = np.array([]) 
        predicts_split = np.array([])
        
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

            accuracy_split = np.append(accuracy_split, accuracy)
            predicts_split = np.append(predicts_split, predictions_val)

            print(f'Acuracia do modelo {model} do Fold {idx}: {accuracy}')        

        output = {
            'accuracy': np.mean(accuracy_split) * 100,
            'std': np.std(accuracy_split),
            'predictions': predicts_split
        }
        return output


    def train_feature_combination(self, k, model, dataframe, list_features, size_comb):
        comb_features = np.array(list(combinations(list_features, size_comb)))
        for i in comb_features:
            self.count  = self.count  + 1
            X = dataframe.iloc[:,i]
            print(f'Teste {self.count} -> features Selecionada para o treino: {X.columns}')
            result = self.cross_validate_balancead(k=k,  model=model, X=X, y=dataframe['labels'].to_frame())
            
            accuracy = result["accuracy"]
            print(f'Accuracy {accuracy} do teste -> {self.count}')
            if accuracy >= 0.7:
                self.dic_result['features'].append(X.columns)
                self.dic_result['accuracy'].append(accuracy)

        return self.dic_result   


    def selector_sequential(self, k, model_estimator, n_features, X, y):
        sfs = SequentialFeatureSelector(
            cv=k, 
            direction = 'forward',
            n_features_to_select = n_features,
            estimator=model_estimator
        )

        sfs.fit(X, y)
        mask_feature = sfs.get_support()
        return X[X.columns[mask_feature]]


    def train_models(self, X, y, models):
        return {f'{str(m)[:-2]}':self.cross_validate_balancead(k=5, model=m, X=X,  y=y.to_frame()) for m in models} 

    def train_tunning_hyperparameters(self, dataframe, model, parameters, filename, cv=5):  
        dict_output = defaultdict(list)  
        parameters_knn = self.ac.tunning_hyperparameters_knn(dataframe=dataframe, log=False)
        bayes_search = BayesSearchCV(
            model,
            parameters,
            n_iter=32,
            n_jobs=-1,
            cv=cv,
            scoring='accuracy'
        )
        for i in range(len(parameters_knn)):
            print(f'interação {i} -> Metric: {parameters_knn[i]["Metric"]}, Algoritmo: {parameters_knn[i]["algorithm"]}, neighbor: {parameters_knn[i]["neighbors"]}')
            new_df = self.ac.show_outilers(dataframe=dataframe, pred=parameters_knn[i]['outilers'])
            X = new_df.drop(columns=["instrumento", "labels"])
            y = new_df["labels"]
            bayes_search.fit(X, y)

            # salvando os resultados
            dict_output["metric_detected_outiler"].append(parameters_knn[i]["Metric"])
            dict_output["algorithm_detected_outiler"].append(parameters_knn[i]["algorithm"])
            dict_output["neighbors_detected_outiler"].append(parameters_knn[i]["neighbors"])

            print(f'interação {i} - Acuracy models: {bayes_search.best_score_ * 100}')
            dict_output["parametos_models"].append(bayes_search.best_params_)

            dict_output["accuracy_models"].append(bayes_search.best_score_ * 100)

        # preenchendo dataframe de saida
        df_output = pd.DataFrame.from_dict(dict_output)

        # salvando o artefado
        pickle.dump(df_output, open(self.wp.directory_row+filename, 'wb'))
        return df_output    