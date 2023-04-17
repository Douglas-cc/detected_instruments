import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

    
    def show_inlers(self, dataframe, pred):
        outilers_id =[pred[i] == 0 for i in range(pred.shape[0])]
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


    def table_outilers_inst(self, dataframe, pred):
        # fazendo filtragens...
        outilers = self.show_outilers(dataframe, pred)
        inlers = self.show_inlers(dataframe, pred)

        # calcular o tatal de outilers por intrumento
        outilers = outilers.instrumento.value_counts().reset_index()
        outilers = outilers.rename(
            columns={
                    "index":"instrumento", 
                    "instrumento":"outliers"
                }
            )

        # calcular o tatal de inlers por intrumento
        inlers = inlers.instrumento.value_counts().reset_index()
        inlers = inlers.rename(
            columns={
                "index":"instrumento", 
                "instrumento":"inlers"  
            }
        )

        output = outilers.merge(inlers, on='instrumento', how='left')
        return output.append(
            {
                "instrumento": "total", 
                "outliers": sum(output["outliers"]),
                "inlers": sum(output["inlers"])
            }, 
            ignore_index=True
        )
    
  
    def plot_outilers_inst(self, dataframe, pred):
        total_outilers = self.table_outilers_inst(dataframe, pred)
        sns.barplot(x='instrumento', y="outliers", data=total_outilers.iloc[:-1])
        plt.xticks(rotation=90)
        return plt.show()
    

    def matriz_confusion(self, y_validate, predicts, title, labels=None, rename_labels=False):
        matrix = confusion_matrix(y_true=y_validate, y_pred=predicts)
        ax = sns.heatmap(matrix, annot=True, fmt='d')
        ax.set(title=title)

        if rename_labels:
            ax.xaxis.set_ticklabels(labels, rotation = 90)
            ax.yaxis.set_ticklabels(labels, rotation = 90)
        return ax
        

    def plot_shap_tree(self, model, X_train, y_train, size=(8, 8)):
        explainer = shap.TreeExplainer(model=model)
        shap_values_train = explainer.shap_values(X_train, y_train)
        expected_value = explainer.expected_value
    
        waterfall = shap.plots._waterfall.waterfall_legacy(
            expected_value=expected_value[1], 
            shap_values=shap_values_train[1][2].reshape(-1), 
            feature_names=X_train.columns, show=True
        )
        summary_bar = shap.summary_plot(shap_values_train[1], X_train, plot_type="bar", plot_size=size)
        summary_dot = shap.summary_plot(shap_values_train[1], X_train, plot_type="dot", plot_size=size)