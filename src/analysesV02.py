import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


class Analytics:
    def __init__(self):
        self.clf_knn = KNN()
        self.neighbors = [3, 5, 7, 9, 11]                                      
        self.algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        self.metric = ['euclidean', 'manhattan', 'minkowski']
        self.features_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 


    def detected_outilers_knn(self, dataframe, algorithm, neighbors, metric):
        '''
        labels binarias (0: inliers, 1: outliers) de previsão 
        e pontuações discrepantes dos dados
        '''
        X = dataframe.select_dtypes(include=self.features_numerics).drop(columns=['labels'])
        X = dataframe
        y = clf.labels_
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        clf = KNN(n_neighbors=self.neighbors, algorithm=self.algorithm, metric=self.metric)
        clf.fit(X_train)
        
        return y_test_pred         


    def tunning_hyperparameters_knn(self, dataframe):
        result = list()
        for k in range(len(neighbors)):
            for a in range(len(algorithms)):
                for m in range(len(metrics)):
                    
                    pred = detected_outilers_knn(
                        dataframe=dataframe,
                        algorithm=self.algoritmo[a],
                        neighbors=self.neighbors[k], 
                        metric=self.metric [m]
                    )

                    atual = [pred, self.neighbors[k], self.algorithms[a], self.metrics[m]]
                    result.append(atual)

                    print('PARAMETROS')
                    print(f'resultado: {pred} -> n_neighbors: {neighbors[k]} algoritmo: {algorithms[a]} - distancia: {metrics[m]}')
                    print()

        return result


    def graphic_anomalies(self, dataframe, pred, title):
        dataframe_numeric = dataframe.select_dtypes(include=self.features_numerics)
        result_f = TSNE(n_components = 2, random_state=1).fit_transform(dataframe_numeric)
        result_f = pd.DataFrame(result_f)
        color= ['orange' if row == 1 else 'purple' for row in pred]
        plt.title(title)
        return plt.scatter(result_f[0], result_f[1], s=1, c=color)


    def show_outilers(self, dataframe, pred):
        outilers_id =[pred[i] == 1 for i in range(pred.shape[0])]
        return dataframe.iloc[outilers_id,:]        


    def features_corr(self, dataframe, limit_inf, limit_sup):
        df_corr = dataframe[dataframe.columns[:-2]].corr().unstack().reset_index().dropna()
    
        df_corr.rename(
            columns = {
                'level_0': 
                'features_a', 
                'level_1': 'features_b',
                0:'correlacao'
            }, 
            inplace = True
        )

        df_corr = df_corr.query(f'correlacao > {limit_inf} and correlacao < {limit_sup}')
        features_select = df_corr.features_b.unique()
        features = np.concatenate((features_select, ['labels','instrumento']))
        return dataframe[features]        


    def table_outilers_inst(self, dataframe_outilers):
        total_outilers = dataframe_outilers.instrumento.value_counts().reset_index()
        total_outilers = total_outilers.rename(
            columns={
                'index':'nome', 
                'instrumento':'total_inst'
            }
        )
        return total_outilers
  

    def plot_outilers_inst(self, dataframe_outilers):
        total_outilers = table_outilers_inst(dataframe_outilers)
        sns.barplot(x='nome', y="total_inst", data=total_outilers)
        plt.xticks(rotation=90)
        return plt.show()