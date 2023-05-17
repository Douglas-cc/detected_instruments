import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.utils import  class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from src.wrapped import Wrapped
from src.analysesV02 import Analytics

class TrainModels:
    def __init__(self):
        self.le = LabelEncoder()
        self.count = 0
        self.ac = Analytics()
        self.features_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
        self.wp = Wrapped(
            '../data/row/',
            '../data/processed/',
            '../data/files/'
        )
        
    def cross_validate_balancead(self, k, dataframe, y_pred, model, oversampling=False, weight=False, shap=False):                
        # labelEncoder para o y_pred
        dataframe['labels'] = self.le.fit_transform(dataframe[y_pred])
        size_data = dataframe.shape[0]

        # definindo X e Y e tranformando y em series para dataframe de unidimensão
        y = dataframe['labels'].to_frame()
        X = dataframe.select_dtypes(include=self.features_numerics)
        X = X.drop(columns="labels")

        kfold =  StratifiedKFold(n_splits=k) 
        name_model = str(model)
    
        # arrays resultados (talvez depois mudar para algo mais dict(list))
        folds = np.array([])
        accuracy = np.array([])
        predictions = np.array([])
        predictions_cat = np.array([])
        y_validate = np.array([])
        y_validate_cat = np.array([])

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
            y_split_validate = y.iloc[idx_validate, :].values
        
            # validacao SEM oversampling, amostra do mundo real com dados desbalanceados
            predictions_split = model.predict(X_split_validate)
            accuracy_split = accuracy_score(y_split_validate, predictions_split)

            # salvar outputs...
            folds = np.append(folds, idx+1)
            accuracy = np.append(accuracy, accuracy_split)
            predictions = np.append(predictions, predictions_split)
            y_validate = np.append(y_validate, y_split_validate)

            # salvar predições e o y como categorico
            predictions_split_cat = predictions_split.copy()
            y_split_validate_cat = y_split_validate.copy()

            predictions_cat = np.append(predictions_cat, self.le.inverse_transform(predictions_split_cat))
            y_validate_cat = np.append(y_validate_cat, self.le.inverse_transform(y_split_validate_cat))
            
            print(f'Tamanho base: {size_data} - Acuracia do modelo {model} do Fold {idx}: {accuracy_split}')

            # gerar graficos shap 
            if shap:
                self.ac.plot_shap_tree(model=model, X_train=X_split_train, y_train=y_split_train)
        dict_output = {
            'folds':folds,
            'accuracy_folds': accuracy,
            'accuracy_mean': np.mean(accuracy) * 100,
            'std': np.std(accuracy), 
            'predictions': predictions, 
            'y_validate': np.reshape(y_validate, (size_data, )),
            'predictions_cat': predictions_cat, 
            'y_validate_cat': np.reshape(y_validate_cat, (size_data, ))
        }
        return dict_output


    def train_feature_combination(self, k, model, y_pred, dataframe, list_features, size_comb):
        comb_features = np.array(list(combinations(list_features, size_comb)))
        dict_output = defaultdict(list)
        for i in comb_features:
            self.count  = self.count  + 1
            X = dataframe.iloc[:,i]

            print(f'Teste {self.count} -> features Selecionada para o treino: {X.columns}')
            result = self.cross_validate_balancead(k=k,  model=model, dataframe=dataframe, y_pred=y_pred)
            
            accuracy = result["accuracy"]
            print(f'Accuracy {accuracy} do teste -> {self.count}')
            
            if accuracy >= 0.7:
                dict_output['features'].append(X.columns)
                dict_output['accuracy'].append(accuracy)
        return dict_output   


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


    def train_models(self, dataframe, y_pred, models):
        return {f'{str(m)[:-2]}':self.cross_validate_balancead(k=5, model=m, dataframe=dataframe, y_pred=y_pred) for m in models} 


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
            new_df = self.ac.show_inlers(dataframe=dataframe, pred=parameters_knn[i]['outilers'])
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