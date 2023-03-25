import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from sklearn.manifold import TSNE


class Analytics:
    def __init__(self):
        self.clf_knn = KNN()
        self.neighbors = [3, 5, 7, 9, 11]                                      
        self.algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        self.metrics = ['euclidean', 'manhattan', 'minkowski']
        self.features_numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 


    def detected_outilers_knn(self, dataframe, algorithm, neighbors, metric):
        clf = KNN(n_neighbors=neighbors, algorithm=algorithm, metric=metric)
        X = dataframe.select_dtypes(include=self.features_numerics)
        clf.fit(X)       
        pred = clf.labels_   # outlier labels (0 or 1)
        scores = clf.decision_scores_  # outlier scores
        return {"scores":scores, "predictions":pred}
    

    def tunning_hyperparameters_knn(self, dataframe, log=True):
        output = list()
        for k in range(len(self.neighbors)):
            for a in range(len(self.algorithms)):
                for m in range(len(self.metrics)):
                    result = self.detected_outilers_knn(
                        dataframe=dataframe,
                        algorithm=self.algorithms[a],
                        neighbors=self.neighbors[k], 
                        metric=self.metrics[m]
                    )                
                    atual = {
                        "outilers": result["predictions"],
                        "scores": result["scores"],
                        "algorithm": self.algorithms[a],
                        "neighbors": self.neighbors[k],
                        "Metric": self.metrics[m]
                    }
                    output.append(atual)
                    if log:
                        print('PARAMETROS')
                        print(f'Resultados -> n_neighbors: {self.neighbors[k]} algoritmo: {self.algorithms[a]} - distancia: {self.metrics[m]}')
                        print()
        return output


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
        total_outilers = self.table_outilers_inst(dataframe_outilers)
        sns.barplot(x='nome', y="total_inst", data=total_outilers)
        plt.xticks(rotation=90)
        return plt.show()