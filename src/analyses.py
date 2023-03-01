import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from pyod.utils.data import evaluate_print


def detected_outilers_knn(dataframe, algorithm, neighbors, metric):
    '''
    labels binarias (0: inliers, 1: outliers) de previsão 
    e pontuações discrepantes dos dados
    '''
    dataframe = dataframe.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    clf = KNN(n_neighbors=neighbors, algorithm=algorithm, metric=metric)
    clf.fit(dataframe)
    
    X = dataframe
    y = clf.labels_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    y_test_pred = clf.predict(X_test)   # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    #evaluate_print('KNN', y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print('KNN', y_test, y_test_scores)

    return y_test_pred 
    

def tunning_hyperparameters_knn(dataframe, algorithm, neighbors, metric):
    result = list()
    for k in range(len(neighbors)):
        for a in range(len(algorithm)):
            for m in range(len(metric)):
                pred = detected_outilers_knn(
                    dataframe=dataframe,
                    algorithm=algorithm[a],
                    neighbors=neighbors[k], 
                    metric=metric[m]
                )
                
                atual = [pred, neighbors[k], algorithm[a], metric[m]]
                result.append(atual)

                print('PARAMETROS')
                print(f'resultado: {pred} -> n_neighbors: {neighbors[k]} algoritmo: {algorithm[a]} - distancia: {metric[m]}')
                print()

    return result


def graphic_anomalies(dataframe, pred, title):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dataframe_numeric = dataframe.select_dtypes(include=numerics)
    
    result_f = TSNE(n_components = 2, random_state=1).fit_transform(dataframe_numeric)
    result_f = pd.DataFrame(result_f)

    # pred = detected_outilers(dataframe)
    color= ['orange' if row == 1 else 'purple' for row in pred]
    
    plt.title(title)
    return plt.scatter(result_f[0], result_f[1], s=1, c=color)
    

def show_outilers(dataframe, pred):
    outilers_id =[pred[i] == 1 for i in range(pred.shape[0])]
    return dataframe.iloc[outilers_id,:]


def remove_outilers(dataframe, pred):
    outilers_id =[pred[i] == 0 for i in range(pred.shape[0])]
    return dataframe.iloc[outilers_id,:]


def features_corr(dataframe, limit_inf, limit_sup):
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


def table_outilers_inst(dataframe_outilers):
    total_outilers = dataframe_outilers.instrumento.value_counts().reset_index()
    total_outilers = total_outilers.rename(
        columns={
            'index':'nome', 
            'instrumento':'total_inst'
        }
    )
    return total_outilers
  

def plot_outilers_inst(dataframe_outilers):
    total_outilers = table_outilers_inst(dataframe_outilers)
    sns.barplot(x='nome', y="total_inst", data=total_outilers)
    plt.xticks(rotation=90)
    return plt.show()